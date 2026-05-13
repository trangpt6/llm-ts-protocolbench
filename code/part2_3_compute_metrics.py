"""
Compute Part 2 metrics from parsed chat outputs.

Input:
    logs/master-logs/master-log-interactive.csv

Outputs:
    results/part2-interactive-llm-forecasting/part2-metrics.csv
    results/part2-interactive-llm-forecasting/forecast-outputs/<file_id>.csv
    results/part2-interactive-llm-forecasting/execution-logs/<file_id>.txt
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from part2_api import datasets, logging_utils, model_setup, paths

load_dataset = datasets.load_dataset
setup_logger = logging_utils.setup_logger
load_model_setup = model_setup.load_model_setup
DEFAULT_MODEL_SETUP_PATH = paths.DEFAULT_MODEL_SETUP_PATH
LOG_DIR = paths.LOG_DIR
RESULTS_DIR = paths.RESULTS_DIR


BASE_DIR = Path(__file__).resolve().parent.parent
PARSED_MASTER_LOG = BASE_DIR / "logs" / "master-logs" / "master-log-part2-interactive-llm-forecasting.csv"
PART2_RESULTS_DIR = RESULTS_DIR / "part2-interactive-llm-forecasting"
METRICS_PATH = PART2_RESULTS_DIR / "metrics-part2-interactive-llm-forecasting.csv"
FORECAST_OUTPUT_DIR = PART2_RESULTS_DIR / "forecast-outputs"
EXECUTION_LOG_DIR = PART2_RESULTS_DIR / "execution-logs"
SCRIPT_TIMEOUT_SEC = 900

logger = logging.getLogger("part2_compute_metrics")


def main() -> None:
    global logger
    logger = setup_logger("part2-compute-metrics", LOG_DIR / "master-logs")
    if not PARSED_MASTER_LOG.exists():
        raise FileNotFoundError(f"Parsed master log not found: {PARSED_MASTER_LOG}")

    df = pd.read_csv(PARSED_MASTER_LOG)
    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Part2 metrics", unit="run"):
        records.append(process_row(row))

    out_df = pd.DataFrame(records)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    logger.info("Saved metrics: %s rows=%d", METRICS_PATH, len(out_df))
    print(f"Saved metrics -> {METRICS_PATH}")


def process_row(row: pd.Series) -> dict:
    file_id = Path(str(row.get("chat_log_file", ""))).stem
    dataset = str(row.get("dataset", ""))
    scenario = str(row.get("scenario", ""))
    record = {
        "file_id": file_id,
        "dataset": dataset,
        "scenario": scenario,
        "branch": row.get("branch", ""),
        "llm_version": row.get("llm_version", ""),
        "run_id": row.get("run_id", ""),
        "output_type": row.get("t3_output_type", ""),
        "execution_status": "PENDING",
        "forecast_length": "",
        "expected_length": "",
        "mae": np.nan,
        "rmse": np.nan,
        "smape": np.nan,
        "mase": np.nan,
        "r2": np.nan,
        "error": "",
    }

    try:
        setup = load_model_setup(DEFAULT_MODEL_SETUP_PATH, dataset, scenario, "baseline")
        bundle = load_dataset(setup.dataset, setup.target_col, source="auto")
        y_true = bundle.test_target_values()
        record["expected_length"] = len(y_true)

        forecasts = load_forecasts(row, file_id)
        record["forecast_length"] = len(forecasts)
        if len(forecasts) != len(y_true):
            raise ValueError(f"forecast length mismatch: expected {len(y_true)}, got {len(forecasts)}")

        naive_factor = calculate_naive_factor(bundle.train_target_values())
        metrics = calculate_metrics(y_true, forecasts, naive_factor)
        record.update(metrics)
        write_forecast_output(bundle, forecasts, FORECAST_OUTPUT_DIR / f"{file_id}.csv")
        record["execution_status"] = "OK"
    except Exception as exc:
        record["execution_status"] = "FAIL"
        record["error"] = str(exc)
        logger.error("Failed %s: %s", file_id, exc)

    return record


def load_forecasts(row: pd.Series, file_id: str) -> list[float]:
    output_type = str(row.get("t3_output_type", ""))
    if output_type == "LIST":
        return parse_forecast_list(str(row.get("t3_forecast_list", "")))
    if output_type == "SCRIPT":
        script_file = Path(str(row.get("script_file", "")))
        if not script_file.exists():
            raise FileNotFoundError(f"script file not found: {script_file}")
        return execute_script(script_file, file_id)
    raise ValueError(f"unsupported output type: {output_type}")


def execute_script(script_file: Path, file_id: str) -> list[float]:
    EXECUTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXECUTION_LOG_DIR / f"{file_id}.txt"
    completed = subprocess.run(
        [sys.executable, script_file.name],
        cwd=script_file.parent,
        capture_output=True,
        text=True,
        timeout=SCRIPT_TIMEOUT_SEC,
        check=False,
    )
    log_path.write_text(
        "STDOUT\n"
        f"{completed.stdout}\n\n"
        "STDERR\n"
        f"{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"script exited with code {completed.returncode}; see {log_path}")
    return parse_forecast_list(completed.stdout)


def parse_forecast_list(text: str) -> list[float]:
    candidate = extract_last_list_literal(text)
    if not candidate:
        raise ValueError("no Python forecast list found")
    value = ast.literal_eval(candidate)
    if not isinstance(value, list):
        raise ValueError("forecast object is not a list")
    forecasts = []
    for item in value:
        if isinstance(item, (list, tuple, dict)):
            raise ValueError("forecast list is not flat")
        forecasts.append(float(item))
    return forecasts


def extract_last_list_literal(text: str) -> str:
    end = text.rfind("]")
    if end == -1:
        return ""
    start = text.rfind("[", 0, end)
    while start != -1:
        candidate = text[start : end + 1].strip()
        try:
            ast.literal_eval(candidate)
            return candidate
        except Exception:
            start = text.rfind("[", 0, start)
    return ""


def calculate_metrics(y_true: np.ndarray, forecasts: list[float], naive_factor: float | None) -> dict[str, float]:
    y_pred = np.array(forecasts, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        raise ValueError("no finite y_true/y_pred pairs")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom == 0, 0, 2 * np.abs(errors) / denom)) * 100)
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - np.sum(errors**2) / sst) if sst > 0 else np.nan
    mase = float(mae / naive_factor) if naive_factor and np.isfinite(naive_factor) else np.nan
    return {"mae": mae, "rmse": rmse, "smape": smape, "mase": mase, "r2": r2}


def calculate_naive_factor(y_train: np.ndarray) -> float:
    y_train = np.asarray(y_train, dtype=float)
    mask = np.isfinite(y_train)
    if not mask.any():
        raise ValueError("no finite training values to compute naive factor")
    y_train = y_train[mask]
    if len(y_train) < 2:
        raise ValueError("not enough training values to compute naive factor")
    diffs = np.diff(y_train)
    return float(np.mean(np.abs(diffs)))


def write_forecast_output(bundle, forecasts: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bundle.test_df.copy()
    out["forecast"] = forecasts
    out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
