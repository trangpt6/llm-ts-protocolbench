from __future__ import annotations

import csv
import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .api_client import LLMClient
from .datasets import DatasetBundle
from .model_setup import ModelSetup
from .prompts import build_part2_turns
from .settings import MODEL_TRACK_LABELS

logger = logging.getLogger(__name__)


@dataclass
class RunOptions:
    run_summary_path: Path
    turn_metrics_path: Path
    chat_log_dir: Path
    csv_mode: str
    max_csv_rows: int
    delay_between_turns: float
    force: bool = False


class CsvAppendLogger:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict[str, Any]) -> None:
        clean = {field: row.get(field, "") for field in self.fieldnames}
        with self.path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(clean)

    def existing_success_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return {
                row.get("run_key", "")
                for row in reader
                if row.get("status") in {"CHAT_DONE", "OK"} and row.get("run_key")
            }

    def existing_turn_keys(self) -> set[tuple[str, str]]:
        """Return set of (run_key, turn_id) pairs already logged in turn_metrics."""
        if not self.path.exists():
            return set()
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return {
                (row.get("run_key", ""), row.get("turn", ""))
                for row in reader
                if row.get("run_key") and row.get("turn") is not None
            }

MASTER_FIELDS = [
    "timestamp",
    "run_key",
    "dataset",
    "scenario",
    "provider",
    "llm_name",
    "model_id",
    "model_track",
    "model_track_label",
    "fixed_model",
    "fixed_hyperparameters",
    "chat_path",
    "run_id",
    "status",
    "n_turns",
    "error",
]


TURN_FIELDS = [
    "timestamp",
    "run_key",
    "dataset",
    "scenario",
    "provider",
    "llm_name",
    "model_id",
    "model_track",
    "model_track_label",
    "run_id",
    "turn",
    "label",
    "latency_s",
    "success",
    "response_length",
    "prompt_words_est",
    "error",
]


def make_run_key(dataset: str, scenario: str, llm_name: str, model_track: str, run_id: int, fixed_model: str) -> str:
    raw = f"dataset={dataset}|scenario={scenario}|llm={llm_name}|track={model_track}|run={run_id}|model={fixed_model}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{digest}_{dataset}_{scenario}_{llm_name}_{model_track}_run{run_id:02d}"


def run_part2_single(
    client: LLMClient,
    bundle: DatasetBundle,
    setup: ModelSetup,
    run_id: int,
    options: RunOptions,
) -> dict[str, Any]:
    run_summary = CsvAppendLogger(options.run_summary_path, MASTER_FIELDS)
    turn_metrics = CsvAppendLogger(options.turn_metrics_path, TURN_FIELDS)
    track_label = MODEL_TRACK_LABELS[setup.model_track]
    chat_filename = f"Part2-{bundle.name}-{setup.scenario}-{track_label}-{client.display_name}-Run{run_id}.txt"
    chat_path = options.chat_log_dir / chat_filename
    run_key = make_run_key(bundle.name, setup.scenario, client.display_name, track_label, run_id, setup.model_name)

    if run_key in run_summary.existing_success_keys() and not options.force:
        logger.info("Skip completed run from master log: %s", run_key)
        return {"run_key": run_key, "skipped": True}

    system_prompt, turns = build_part2_turns(bundle, setup, options.csv_mode, options.max_csv_rows)
    history = [{"role": "system", "content": system_prompt}]
    chat_log: dict[str, Any] = {
        "meta": _run_meta(run_key, client, bundle, setup, run_id, chat_path),
        "system_prompt": system_prompt,
        "turns": [],
        "status": "running",
    }
    error = ""

    try:
        existing_turns = turn_metrics.existing_turn_keys()
        for turn in turns:
            messages = history + [{"role": "user", "content": turn.content}]
            response = _call_turn(client, messages, turn, turn_metrics, existing_turns, run_key, bundle, setup, run_id)
            history.extend([{"role": "user", "content": turn.content}, {"role": "assistant", "content": response}])
            chat_log["turns"].append(
                {"turn": turn.turn_id, "label": turn.label, "user": turn.content, "assistant": response}
            )
            if options.delay_between_turns and turn.turn_id < turns[-1].turn_id:
                time.sleep(options.delay_between_turns)

        status = "CHAT_DONE"
    except Exception as exc:
        status = "FAIL"
        error = str(exc)
        logger.error("Run failed %s: %s", run_key, error)

    chat_log["status"] = status
    chat_log["error"] = error
    _write_chat_text(chat_path, chat_log, client.display_name)
    run_summary.log(
        {
            "timestamp": datetime.now().isoformat(),
            "run_key": run_key,
            "dataset": bundle.name,
            "scenario": setup.scenario,
            "provider": client.provider,
            "llm_name": client.display_name,
            "model_id": client.model_id,
            "model_track": setup.model_track,
            "model_track_label": track_label,
            "fixed_model": setup.model_name,
            "fixed_hyperparameters": setup.hyperparameters_text,
            "chat_path": str(chat_path),
            "run_id": run_id,
            "status": status,
            "n_turns": len(chat_log["turns"]),
            "error": error,
        }
    )
    return {"run_key": run_key, "status": status, "chat_path": chat_path, "error": error}


def _call_turn(
    client: LLMClient,
    messages: list[dict[str, str]],
    turn,
    turn_metrics: CsvAppendLogger,
    existing_turns: set[tuple[str, str]],
    run_key: str,
    bundle: DatasetBundle,
    setup: ModelSetup,
    run_id: int,
) -> str:
    # Skip logging if (run_key, turn_id) is already recorded
    already_logged = (run_key, str(turn.turn_id)) in existing_turns

    started = time.time()
    try:
        response = client.chat(messages)
        latency = time.time() - started
        if not already_logged:
            turn_metrics.log({
                "timestamp": datetime.now().isoformat(),
                "run_key": run_key,
                "dataset": bundle.name,
                "scenario": setup.scenario,
                "provider": client.provider,
                "llm_name": client.display_name,
                "model_id": client.model_id,
                "model_track": setup.model_track,
                "model_track_label": MODEL_TRACK_LABELS[setup.model_track],
                "run_id": run_id,
                "turn": turn.turn_id,
                "label": turn.label,
                "latency_s": round(latency, 3),
                "success": True,
                "response_length": len(response),
                "prompt_words_est": sum(len(message["content"].split()) for message in messages),
                "error": "",
            })
        return response
    except Exception as exc:
        latency = time.time() - started
        if not already_logged:
            turn_metrics.log({
                "timestamp": datetime.now().isoformat(),
                "run_key": run_key,
                "dataset": bundle.name,
                "scenario": setup.scenario,
                "provider": client.provider,
                "llm_name": client.display_name,
                "model_id": client.model_id,
                "model_track": setup.model_track,
                "model_track_label": MODEL_TRACK_LABELS[setup.model_track],
                "run_id": run_id,
                "turn": turn.turn_id,
                "label": turn.label,
                "latency_s": round(latency, 3),
                "success": False,
                "response_length": 0,
                "prompt_words_est": sum(len(message["content"].split()) for message in messages),
                "error": str(exc),
            })
        raise


def _run_meta(run_key: str, client: LLMClient, bundle: DatasetBundle, setup: ModelSetup, run_id: int, chat_path: Path) -> dict[str, Any]:
    return {
        "run_key": run_key,
        "dataset": bundle.name,
        "dataset_source": bundle.source,
        "target_col": bundle.target_col,
        "scenario": setup.scenario,
        "scenario_description": setup.scenario_description,
        "horizon_or_block": setup.horizon_or_block,
        "provider": client.provider,
        "llm_name": client.display_name,
        "model_id": client.model_id,
        "model_track": setup.model_track,
        "fixed_model": setup.model_name,
        "fixed_hyperparameters": setup.hyperparameters,
        "run_id": run_id,
        "chat_path": str(chat_path),
    }


def _write_chat_text(path: Path, chat_log: dict[str, Any], llm_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = chat_log.get("meta", {})
    parts = [
        "Part 2 Interactive Chat Log",
        "",
        f"Dataset: {meta.get('dataset', '')}",
        f"Scenario: {meta.get('scenario', '')}",
        f"Forecast model type: {MODEL_TRACK_LABELS.get(meta.get('model_track', ''), meta.get('model_track', ''))}",
        f"Fixed forecasting model: {meta.get('fixed_model', '')}",
        f"LLM: {llm_name}",
        f"Provider: {meta.get('provider', '')}",
        f"Run: {meta.get('run_id', '')}",
        "",
        "===== SYSTEM INSTRUCTION =====",
        "",
    ]
    system_content = str(chat_log.get("system_prompt", ""))
    turns = chat_log.get("turns", [])
    parts.append(system_content)
    for turn in turns:
        turn_no = turn.get("turn", "")
        label = turn.get("label", "")
        parts.extend(
            [
                "",
                f"===== TURN {turn_no} USER ({label}) =====",
                "",
                str(turn.get("user", "")),
                "",
                f"===== TURN {turn_no} ASSISTANT ({llm_name}) =====",
                "",
                str(turn.get("assistant", "")),
            ]
        )
    path.write_text("\n".join(parts), encoding="utf-8")
