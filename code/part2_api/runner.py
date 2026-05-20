from __future__ import annotations

import ast
import csv
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .api_client import LLMClient, LLMSystemError
from .datasets import DatasetBundle
from .model_setup import ModelSetup
from .prompts import build_part2_turns
from .settings import (
    LEGACY_SUCCESS_STATUSES,
    ILINET_TURN_MAX_TOKENS,
    ILINET_TURN_TIMEOUT_SECONDS,
    MAX_SCRIPT_CHARS,
    MODEL_TRACK_LABELS,
    SUCCESS_STATUSES,
    ETTH1_TURN_TIMEOUT_SECONDS,
    TURN_3_MAX_TOKENS_BASE,
    TURN_3_MAX_TOKENS_CAP,
    TURN_3_MAX_TOKENS_PER_ITEM,
    TURN_MAX_TOKENS_DEFAULTS,
    TURN_TIMEOUT_SECONDS,
)

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
        # SUCCESS_STATUSES are the canonical new codes; LEGACY_SUCCESS_STATUSES
        # lets old "CHAT_DONE"/"OK" rows serve as skip-checkpoints too.
        # INVALID_LIST_LENGTH was an old strict status for parseable lists with
        # wrong length; those runs are still complete under the benchmark rule.
        all_success = SUCCESS_STATUSES | LEGACY_SUCCESS_STATUSES | {"INVALID_LIST_LENGTH"}
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return {
                row.get("run_key", "")
                for row in reader
                if row.get("status") in all_success and row.get("run_key")
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
    # New fields appended at end for backward compatibility with existing CSVs.
    "final_output_type",
    "expected_forecast_length",
    "actual_output_length",
    # execution_success: API call succeeded AND Turn 3 returned a parseable/valid format.
    #   True  → status in SUCCESS_STATUSES (OK_FORECAST_LIST or OK_FORECAST_SCRIPT)
    #   False / blank → system error or model format failure
    # protocol_compliant: output satisfies the full benchmark protocol (format + constraints).
    #   True  → valid list with correct length, OR structurally valid script
    #   False → valid format but wrong list length, or any format/structure violation
    #   blank → Turn 3 was not reached (system failure before or during the turn)
    # validation_passed: alias for protocol_compliant; kept for backward compatibility.
    "execution_success",
    "protocol_compliant",
    "validation_passed",
    "length_mismatch",
    "validation_warning",
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


# ---------------------------------------------------------------------------
# Turn-3 helpers
# ---------------------------------------------------------------------------

def _expected_forecast_length(bundle: DatasetBundle, setup: ModelSetup) -> int:
    return len(bundle.test_df)


def _get_turn_policy(turn_id: int, bundle: DatasetBundle, setup: ModelSetup) -> dict[str, Any]:
    """Return {"max_tokens": int, "timeout": int|None} for a given turn."""
    if bundle.name == "ILINet":
        timeout_by_turn = ILINET_TURN_TIMEOUT_SECONDS
    elif bundle.name == "ETTh1":
        timeout_by_turn = ETTH1_TURN_TIMEOUT_SECONDS
    else:
        timeout_by_turn = TURN_TIMEOUT_SECONDS
    timeout = timeout_by_turn.get(turn_id)
    if turn_id == 3:
        if bundle.name == "ILINet":
            max_tokens = ILINET_TURN_MAX_TOKENS[3]
        else:
            expected_len = _expected_forecast_length(bundle, setup)
            max_tokens = min(
                TURN_3_MAX_TOKENS_CAP,
                TURN_3_MAX_TOKENS_BASE + expected_len * TURN_3_MAX_TOKENS_PER_ITEM,
            )
    else:
        max_tokens = (
            ILINET_TURN_MAX_TOKENS.get(turn_id, TURN_MAX_TOKENS_DEFAULTS.get(turn_id, 1200))
            if bundle.name == "ILINet"
            else TURN_MAX_TOKENS_DEFAULTS.get(turn_id, 1200)
        )
    return {"max_tokens": max_tokens, "timeout": timeout}


def _try_parse_forecast_list(text: str) -> list | None:
    """Return the parsed list if *text* (stripped) is a valid flat numeric list, else None.

    Rules: must start with '[', parse via ast.literal_eval, be a flat list,
    every element castable to float, no booleans.
    """
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    try:
        value = ast.literal_eval(stripped)
    except Exception:
        return None
    if not isinstance(value, list):
        return None
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, (list, tuple, dict)):
            return None
        try:
            float(item)
        except (TypeError, ValueError):
            return None
    return value


def _check_script_format(text: str) -> tuple[bool, str]:
    """Minimal structural checks for a Turn-3 Python script response.

    Returns (is_valid, error_message).
    """
    stripped = text.strip()
    if not stripped:
        return False, "Script is empty."
    if stripped.startswith("```"):
        return False, "Script starts with a markdown code fence."
    if "```" in stripped:
        return False, "Script contains a markdown code fence."
    python_patterns = [
        r"^\s*(import|from)\s+\w+",
        r"^\s*def\s+\w+\s*\(",
        r"^\s*class\s+\w+",
        r"^\s*\w+\s*=\s*",
        r"^\s*for\s+\w+",
        r"^\s*if\s+",
    ]
    if not any(re.search(p, stripped, re.MULTILINE) for p in python_patterns):
        return False, "Response does not appear to be Python code."
    # Accept any variable name that contains a common forecast/prediction keyword:
    # forecasts, forecast_list, forecast_values, predictions, predicted_values, preds, …
    _FORECAST_TERMS = ("forecast", "predict", "preds")
    has_print_forecasts = any(
        re.search(r"print\s*\(", line) and any(t in line.lower() for t in _FORECAST_TERMS)
        for line in stripped.splitlines()
    )
    non_empty_lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    last_line_references_forecasts = bool(
        non_empty_lines
        and any(t in non_empty_lines[-1].lower() for t in _FORECAST_TERMS)
    )
    if not (has_print_forecasts or last_line_references_forecasts):
        return False, "Script does not contain a print or final reference to a forecast/prediction variable."
    return True, ""


def _validate_turn3_response(text: str, bundle: DatasetBundle, setup: ModelSetup) -> dict[str, Any]:
    """Classify a raw Turn-3 response and return a validation result dict.

    Two distinct concepts are captured:

    execution_success (not in this dict — derived by caller as status in SUCCESS_STATUSES):
        True when the API call completed and Turn 3 returned a parseable/valid format,
        regardless of whether the output fully satisfies the benchmark protocol.

    protocol_compliant (returned in this dict):
        True when the output satisfies ALL benchmark constraints:
          - list: parseable AND length == expected_length
          - script: passes all structural checks
        False on any format or constraint violation (including list length mismatch).
        Blank/None when Turn 3 was not reached (system failure).

    Return keys:
        status              – one of the benchmark status codes
        error               – short human-readable description (empty on execution success)
        output_type         – "error" | "list" | "script" | "invalid"
        expected_length     – int
        actual_length       – int or None
        length_mismatch     – bool (True = length differs from expected) or None when not applicable
        validation_warning  – diagnostic string; empty when no issue
        protocol_compliant  – bool; see above
        normalized_response – the original text unchanged
    """
    expected_length = _expected_forecast_length(bundle, setup)
    stripped = text.strip()

    result: dict[str, Any] = {
        "status": "INVALID_OUTPUT",
        "error": "",
        "output_type": "invalid",
        "expected_length": expected_length,
        "actual_length": None,
        "length_mismatch": None,
        "validation_warning": "",
        "protocol_compliant": False,
        "normalized_response": text,
    }

    # Rule 1: literal "ERROR"
    if stripped == "ERROR":
        result.update({
            "status": "MODEL_ERROR_LITERAL",
            "error": "Model returned literal ERROR at Turn 3.",
            "output_type": "error",
            "protocol_compliant": False,
        })
        return result

    # Rule 2: flat numeric Python list — execution succeeds whenever parseable.
    # Length mismatch is a protocol violation: protocol_compliant=False, but status is still
    # OK_FORECAST_LIST (execution success). No retry is triggered by a length mismatch.
    parsed_list = _try_parse_forecast_list(stripped)
    if parsed_list is not None:
        actual_length = len(parsed_list)
        mismatch = actual_length != expected_length
        result.update({
            "status": "OK_FORECAST_LIST",
            "error": "",
            "output_type": "list",
            "actual_length": actual_length,
            "length_mismatch": mismatch,
            "validation_warning": (
                "INVALID_LIST_LENGTH"
                if mismatch else ""
            ),
            "protocol_compliant": not mismatch,
        })
        return result

    # Rule 3: Python script
    if len(stripped) > MAX_SCRIPT_CHARS:
        result.update({
            "status": "INVALID_SCRIPT_TOO_LONG",
            "error": f"Script length {len(stripped)} chars exceeds MAX_SCRIPT_CHARS={MAX_SCRIPT_CHARS}.",
            "output_type": "script",
            "protocol_compliant": False,
        })
        return result

    script_valid, script_error = _check_script_format(stripped)
    if script_valid:
        result.update({
            "status": "OK_FORECAST_SCRIPT",
            "error": "",
            "output_type": "script",
            "length_mismatch": None,
            "validation_warning": "",
            "protocol_compliant": True,
        })
    else:
        result.update({
            "status": "INVALID_SCRIPT_FORMAT",
            "error": script_error,
            "output_type": "invalid",
            "protocol_compliant": False,
        })
    return result


def _map_exception_to_system_status(exc: Exception) -> str:
    """Map any exception to the closest system status code."""
    if isinstance(exc, LLMSystemError):
        return exc.system_status
    err = str(exc).lower()
    from .api_client import TIMEOUT_MARKERS, RATE_LIMIT_MARKERS, SERVER_ERROR_MARKERS
    if any(m in err for m in TIMEOUT_MARKERS):
        return "SYSTEM_TIMEOUT"
    if any(m in err for m in RATE_LIMIT_MARKERS):
        return "SYSTEM_RATE_LIMIT"
    if any(m in err for m in SERVER_ERROR_MARKERS):
        return "SYSTEM_API_ERROR"
    return "SYSTEM_REQUEST_FAILED"


# ---------------------------------------------------------------------------

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
        "error": "",
    }

    # Defaults — overwritten by Turn-3 validator result or caught exceptions.
    status = "SYSTEM_REQUEST_FAILED"
    error = ""
    t3_validation: dict[str, Any] = {}

    try:
        existing_turns = turn_metrics.existing_turn_keys()
        for turn in turns:
            messages = history + [{"role": "user", "content": turn.content}]

            if turn.turn_id == 3:
                # Turn 3: validate output; model failures do NOT raise exceptions.
                response, status, error, t3_validation = _call_and_validate_turn3(
                    client, messages, turn, turn_metrics, existing_turns,
                    run_key, bundle, setup, run_id,
                )
            else:
                # Turns 0-2: propagate system/API failures upward.
                policy = _get_turn_policy(turn.turn_id, bundle, setup)
                response = _call_turn(
                    client, messages, turn, turn_metrics, existing_turns,
                    run_key, bundle, setup, run_id,
                    max_tokens=policy["max_tokens"], timeout=policy["timeout"],
                )

            history.extend([
                {"role": "user", "content": turn.content},
                {"role": "assistant", "content": response},
            ])
            chat_log["turns"].append(
                {"turn": turn.turn_id, "label": turn.label, "user": turn.content, "assistant": response}
            )
            if options.delay_between_turns and turn.turn_id < turns[-1].turn_id:
                time.sleep(options.delay_between_turns)

    except LLMSystemError as exc:
        status = exc.system_status
        error = str(exc)
        logger.error("Run failed %s: [%s] %s", run_key, status, error)
    except Exception as exc:
        status = "SYSTEM_REQUEST_FAILED"
        error = str(exc)
        logger.error("Run failed %s: %s", run_key, error)

    chat_log["status"] = status
    chat_log["error"] = error
    _write_chat_text(chat_path, chat_log, client.display_name)

    run_summary.log({
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
        "final_output_type": t3_validation.get("output_type", ""),
        "expected_forecast_length": t3_validation.get("expected_length", ""),
        "actual_output_length": (
            "" if t3_validation.get("actual_length") is None
            else t3_validation["actual_length"]
        ),
        # execution_success: API reached Turn 3 and returned a parseable/valid format.
        "execution_success": (status in SUCCESS_STATUSES) if t3_validation else "",
        # protocol_compliant: output satisfies ALL benchmark constraints (format + length).
        # False for OK_FORECAST_LIST with wrong length; blank when Turn 3 was not reached.
        "protocol_compliant": t3_validation.get("protocol_compliant", "") if t3_validation else "",
        # validation_passed: alias for protocol_compliant (backward compat field name).
        "validation_passed": t3_validation.get("protocol_compliant", "") if t3_validation else "",
        "length_mismatch": (
            t3_validation.get("length_mismatch", "")
            if t3_validation and t3_validation.get("length_mismatch") is not None
            else ""
        ),
        "validation_warning": t3_validation.get("validation_warning", "") if t3_validation else "",
    })
    return {"run_key": run_key, "status": status, "chat_path": chat_path, "error": error}


# OpenRouter reasoning tokens are billed as output tokens. Turns 0-2 must
# explicitly use effort="none" to control cost; Turn 3 is the only
# reasoning-enabled OpenRouter turn.
_OPENROUTER_REASONING_NONE = {"reasoning": {"effort": "none", "exclude": True}}
_OPENROUTER_REASONING_HIGH = {"reasoning": {"effort": "high", "exclude": True}}
_OPENROUTER_REASONING_CLAUDE_TURN3 = {"reasoning": {"effort": "medium", "exclude": True}}

_OPENROUTER_REASONING: dict[str, dict[str, dict[str, Any]]] = {
    "openai/gpt-5.5": {
        "turns_0_2": _OPENROUTER_REASONING_NONE,
        "turn_3": _OPENROUTER_REASONING_HIGH,
    },
    "anthropic/claude-opus-4.7": {
        "turns_0_2": _OPENROUTER_REASONING_NONE,
        "turn_3": _OPENROUTER_REASONING_CLAUDE_TURN3,
    },
    "x-ai/grok-4.3": {
        "turns_0_2": _OPENROUTER_REASONING_NONE,
        "turn_3": _OPENROUTER_REASONING_HIGH,
    },
}

# Beeknoee (OpenAI-compatible gateway) thinking control for Anthropic models.
# Claude Opus 4.7: keep turns 0-2 without thinking; enable adaptive thinking on turn 3.
_BEEKNOEE_THINKING_DISABLED = {"thinking": {"type": "disabled"}}
_BEEKNOEE_THINKING_ADAPTIVE = {"thinking": {"type": "adaptive"}}

_BEEKNOEE_THINKING: dict[str, dict[str, dict[str, Any]]] = {
    "claude-opus-4-7": {
        "turns_0_2": _BEEKNOEE_THINKING_DISABLED,
        "turn_3": _BEEKNOEE_THINKING_ADAPTIVE,
    },
    "gpt-5.5": {
        "turns_0_2": {"reasoning": {"effort": "none"}},
        "turn_3": {"reasoning": {"effort": "high"}},
    },
}

_XAI_REASONING: dict[str, dict[str, dict[str, Any]]] = {
    "grok-4-3": {
        "turns_0_2": {"reasoning": {"effort": "none"}},
        "turn_3": {"reasoning": {"effort": "medium"}},
    },
}

# MegaLLM (OpenAI-compatible) reasoning control.
# We explicitly disable reasoning on turns 0-2 to reduce cost and only
# enable it on turn 3 for forecast/code generation quality.
_MEGALLM_REASONING_NONE = {"reasoning": {"effort": "none"}}
_MEGALLM_REASONING_HIGH = {"reasoning": {"effort": "high"}}
_MEGALLM_REASONING_MEDIUM = {"reasoning": {"effort": "medium"}}

_MEGALLM_REASONING: dict[str, dict[str, dict[str, Any]]] = {
    "gpt-5.5": {
        "turns_0_2": _MEGALLM_REASONING_NONE,
        "turn_3": _MEGALLM_REASONING_MEDIUM,
    },
    "gemini-3.1-pro-preview": {
        "turns_0_2": _MEGALLM_REASONING_NONE,
        "turn_3": _MEGALLM_REASONING_HIGH,
    },
    "moonshotai/kimi-k2.6": {
        "turns_0_2": _MEGALLM_REASONING_NONE,
        "turn_3": _MEGALLM_REASONING_MEDIUM,
    }
}

_LLMGATE_THINKING: dict[str, dict[str, dict[str, Any]]] = {
    "claude-opus-4-7": {
        "turns_0_2": {"thinking": {"type": "disabled"}},
        "turn_3": {"thinking": {"type": "adaptive"}},
    },
    "gpt-5.5": {
        "turns_0_2": {"reasoning": {"effort": "none"}},
        "turn_3": {"reasoning": {"effort": "high"}},
    },
}

def _get_moonshot_turn_config(turn_id: int) -> tuple[dict[str, Any], float]:
    if turn_id == 3:
        return {"thinking": {"type": "enabled"}}, 1.0
    return {"thinking": {"type": "disabled"}}, 0.6

def _get_llmgate_thinking_config(model_id: str, turn_id: int) -> dict | None:
    configs = _LLMGATE_THINKING.get(model_id)
    if configs is None:
        return None
    config = configs["turn_3"] if turn_id == 3 else configs["turns_0_2"]
    return dict(config)

def _get_beeknoee_thinking_config(model_id: str, turn_id: int) -> dict | None:
    """Return Beeknoee thinking extra_params for *model_id* at *turn_id*, or None if not mapped."""
    configs = _BEEKNOEE_THINKING.get(model_id)
    if configs is None:
        return None
    config = configs["turn_3"] if turn_id == 3 else configs["turns_0_2"]
    return dict(config)

def _get_xai_reasoning_config(model_id: str, turn_id: int) -> dict | None:
    """Return XAI reasoning extra_params for *model_id* at *turn_id*, or None if not mapped."""
    configs = _XAI_REASONING.get(model_id)
    if configs is None:
        return None
    config = configs["turn_3"] if turn_id == 3 else configs["turns_0_2"]
    return {"reasoning": dict(config["reasoning"])}

def _get_megallm_reasoning_config(model_id: str, turn_id: int) -> dict | None:
    """Return MegaLLM reasoning extra_params for *model_id* at *turn_id*, or None if not mapped."""
    configs = _MEGALLM_REASONING.get(model_id)
    if configs is None:
        return None
    config = configs["turn_3"] if turn_id == 3 else configs["turns_0_2"]
    return {"reasoning": dict(config["reasoning"])}

def _get_openrouter_reasoning_config(model_id: str, turn_id: int) -> dict | None:
    """Return OpenRouter reasoning extra_params for *model_id* at *turn_id*, or None if not mapped."""
    configs = _OPENROUTER_REASONING.get(model_id)
    if configs is None:
        return None
    config = configs["turn_3"] if turn_id == 3 else configs["turns_0_2"]
    return {"reasoning": dict(config["reasoning"])}


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
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> str:
    """Call the LLM for turns 0-2 and log metrics. Raises on any API failure."""
    already_logged = (run_key, str(turn.turn_id)) in existing_turns
    temperature: float | None = None
    # Turns 0-2 only need short structured answers; minimise/disable reasoning.
    if client.provider == "deepseek":
        turn_extra_params: dict | None = {"thinking": {"type": "disabled"}}
    elif client.provider == "moonshot":
        turn_extra_params, temperature = _get_moonshot_turn_config(turn.turn_id)
    elif client.provider == "llmgate":
        max_tokens = max(max_tokens or 0, 16384)
        if client.model_id == "kimi-k2.6":
            turn_extra_params, temperature = _get_moonshot_turn_config(turn.turn_id)
        else:
            turn_extra_params = _get_llmgate_thinking_config(client.model_id, turn.turn_id)
    elif client.provider == "xai":
        turn_extra_params = _get_xai_reasoning_config(client.model_id, turn.turn_id)
    elif client.provider.startswith("openrouter"):
        turn_extra_params = _get_openrouter_reasoning_config(client.model_id, turn.turn_id)
    elif client.provider == "megallm":
        if client.model_id == "moonshotai/kimi-k2.6":
            turn_extra_params, temperature = _get_moonshot_turn_config(turn.turn_id)
        else:
            turn_extra_params = _get_megallm_reasoning_config(client.model_id, turn.turn_id)
    elif client.provider == "beeknoee":
        if client.model_id == "kimi-k2.6":
            max_tokens = max(max_tokens or 0, 16384)
            turn_extra_params, temperature = _get_moonshot_turn_config(turn.turn_id)
        else:
            turn_extra_params = _get_beeknoee_thinking_config(client.model_id, turn.turn_id)
    else:
        turn_extra_params = None
    started = time.time()
    try:
        response = client.chat(messages, max_tokens=max_tokens, timeout=timeout, extra_params=turn_extra_params, temperature=temperature)
        if not response or not response.strip():
            raise RuntimeError(
                f"Empty model response at turn {turn.turn_id} (provider={client.provider})."
            )
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
                "prompt_words_est": sum(len(m["content"].split()) for m in messages),
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
                "prompt_words_est": sum(len(m["content"].split()) for m in messages),
                "error": str(exc),
            })
        raise


def _call_and_validate_turn3(
    client: LLMClient,
    messages: list[dict[str, str]],
    turn,
    turn_metrics: CsvAppendLogger,
    existing_turns: set[tuple[str, str]],
    run_key: str,
    bundle: DatasetBundle,
    setup: ModelSetup,
    run_id: int,
) -> tuple[str, str, str, dict[str, Any]]:
    """Call the LLM for Turn 3, validate the response, and log metrics.

    Returns (response_text, status, error, validation_dict).
    Never raises – model validation failures and system errors are both
    represented as status codes in the return value.
    """
    already_logged = (run_key, str(turn.turn_id)) in existing_turns
    policy = _get_turn_policy(3, bundle, setup)
    temperature: float | None = None
    if client.provider == "openrouter_gemini":
        policy["max_tokens"] = 16384
        if policy["timeout"] is None or policy["timeout"] < 180:
            policy["timeout"] = 180
    # Turn 3 requires reasoning quality for forecast/code generation; enable thinking.
    elif client.provider == "moonshot" or client.model_id == ("kimi-k2.6" or "moonshotai/kimi-k2.6"):
        turn_extra_params, temperature = _get_moonshot_turn_config(3)
        policy["max_tokens"] = 16384
        if policy["timeout"] is None or policy["timeout"] < 180:
            policy["timeout"] = 180
    elif client.provider == "llmgate":
        policy["max_tokens"] = 16384
        if policy["timeout"] is None or policy["timeout"] < 180:
            policy["timeout"] = 180
        turn_extra_params = _get_llmgate_thinking_config(client.model_id, 3)
    elif client.provider == "deepseek":
        turn_extra_params: dict | None = {"thinking": {"type": "enabled"}}
    elif client.provider == "xai":
        turn_extra_params = _get_xai_reasoning_config(client.model_id, 3)
    elif client.provider.startswith("openrouter"):
        turn_extra_params = _get_openrouter_reasoning_config(client.model_id, 3)
    elif client.provider == "megallm":
        if client.model_id == "gemini-3.1-pro-preview":
            turn_extra_params = _get_megallm_reasoning_config(client.model_id, 3)
            policy["max_tokens"] = 16384
            if policy["timeout"] is None or policy["timeout"] < 180:
                policy["timeout"] = 180
        else:
            turn_extra_params = _get_megallm_reasoning_config(client.model_id, 3)
    elif client.provider == "beeknoee":
        turn_extra_params = _get_beeknoee_thinking_config(client.model_id, 3)
    else:
        turn_extra_params = None
    started = time.time()
    response = ""
    status = "SYSTEM_REQUEST_FAILED"
    error = ""
    validation: dict[str, Any] = {}

    try:
        response = client.chat(messages, max_tokens=policy["max_tokens"], timeout=policy["timeout"], extra_params=turn_extra_params, temperature=temperature)
        latency = time.time() - started

        validation = _validate_turn3_response(response, bundle, setup)
        status = validation["status"]
        error = validation.get("error", "")
        turn_success = status in SUCCESS_STATUSES

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
                "success": turn_success,
                "response_length": len(response),
                "prompt_words_est": sum(len(m["content"].split()) for m in messages),
                "error": error,
            })

    except LLMSystemError as exc:
        latency = time.time() - started
        status = exc.system_status
        error = str(exc)
        logger.error("Turn 3 system error for %s: [%s] %s", run_key, status, error)
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
                "prompt_words_est": sum(len(m["content"].split()) for m in messages),
                "error": error,
            })

    except Exception as exc:
        latency = time.time() - started
        status = _map_exception_to_system_status(exc)
        error = str(exc)
        logger.error("Turn 3 unexpected error for %s: [%s] %s", run_key, status, error)
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
                "prompt_words_est": sum(len(m["content"].split()) for m in messages),
                "error": error,
            })

    return response, status, error, validation


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
        # Run Status and Run Error allow the parser to read the outcome without
        # consulting the run-summary CSV separately.
        f"Run Status: {chat_log.get('status', '')}",
        f"Run Error: {chat_log.get('error', '')}",
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
