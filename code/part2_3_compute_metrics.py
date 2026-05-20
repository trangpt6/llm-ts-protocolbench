"""Execute parsed Part 2 outputs, compute metrics, and add DM tests.

Inputs:
    logs/master-logs/master-log-part2-interactive-llm-forecasting.csv

Outputs:
    results/part2-interactive-llm-forecasting/metrics-part2-interactive-llm-forecasting.csv
    results/part2-interactive-llm-forecasting/diebold-mariano-part2-interactive-llm-forecasting.csv
    results/part2-interactive-llm-forecasting/forecast-outputs/<file_id>.csv
    results/part2-interactive-llm-forecasting/execution-logs/<file_id>.txt
"""

from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
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
DM_RESULTS_PATH = PART2_RESULTS_DIR / "diebold-mariano-part2-interactive-llm-forecasting.csv"
PARTIAL_METRICS_PATH = PART2_RESULTS_DIR / "metrics-part2-interactive-llm-forecasting.partial.csv"
FORECAST_OUTPUT_DIR = PART2_RESULTS_DIR / "forecast-outputs"
EXECUTION_LOG_DIR = PART2_RESULTS_DIR / "execution-logs"
SCRIPT_TIMEOUT_SEC = 900
DATASET_SCRIPT_TIMEOUT_SEC = {
    "Temperature": 120,
}
RERUN_FAILED_SCRIPTS = False
BRANCH_TO_TRACK = {
    "Base": "baseline",
    "ChalML": "challenger_ml",
    "ChalDL": "challenger_dl",
}

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
        if len(records) % 25 == 0:
            write_partial_metrics(records)

    out_df = pd.DataFrame(records)
    write_partial_metrics(records)
    out_df, dm_df = add_diebold_mariano_results(out_df)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    dm_df.to_csv(DM_RESULTS_PATH, index=False, encoding="utf-8-sig")
    logger.info("Saved metrics: %s rows=%d", METRICS_PATH, len(out_df))
    logger.info("Saved DM results: %s rows=%d", DM_RESULTS_PATH, len(dm_df))
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved DM results -> {DM_RESULTS_PATH}")


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
        "source_run_status": row.get("run_status", ""),
        "source_run_error": row.get("run_error", ""),
        "execution_status": "PENDING",
        "forecast_length": "",
        "expected_length": "",
        "forecast_output_file": "",
        "execution_log_file": "",
        "mae": np.nan,
        "rmse": np.nan,
        "smape": np.nan,
        "mase": np.nan,
        "r2": np.nan,
        "dm_reference_file_id": "",
        "dm_reference_branch": "",
        "dm_loss": "",
        "dm_lag": np.nan,
        "dm_n": np.nan,
        "dm_mean_loss_diff": np.nan,
        "dm_statistic": np.nan,
        "dm_p_value": np.nan,
        "dm_better_than_base": None,
        "error": "",
    }
    if str(row.get("t3_output_type", "")) == "SCRIPT":
        record["execution_log_file"] = to_repo_relative_path(EXECUTION_LOG_DIR / f"{file_id}.txt")

    try:
        setup = load_model_setup(
            DEFAULT_MODEL_SETUP_PATH,
            dataset,
            scenario,
            branch_to_track(str(row.get("branch", ""))),
        )
        bundle = load_dataset(setup.dataset, setup.target_col, source="auto")
        y_true = bundle.test_target_values()
        record["expected_length"] = len(y_true)

        forecast_path = FORECAST_OUTPUT_DIR / f"{file_id}.csv"
        if forecast_path.exists():
            forecasts = load_existing_forecast_output(forecast_path)
        else:
            forecasts = load_forecasts(row, file_id, bundle, dataset)
        record["forecast_length"] = len(forecasts)
        if len(forecasts) != len(y_true):
            raise ValueError(f"forecast length mismatch: expected {len(y_true)}, got {len(forecasts)}")

        naive_factor = calculate_naive_factor(bundle.train_target_values())
        metrics = calculate_metrics(y_true, forecasts, naive_factor)
        record.update(metrics)
        write_forecast_output(bundle, forecasts, forecast_path)
        record["forecast_output_file"] = to_repo_relative_path(forecast_path)
        record["execution_status"] = "OK"
    except Exception as exc:
        record["execution_status"] = "FAIL"
        record["error"] = str(exc)
        logger.error("Failed %s: %s", file_id, exc)

    return record


def write_partial_metrics(records: list[dict]) -> None:
    PARTIAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(PARTIAL_METRICS_PATH, index=False, encoding="utf-8-sig")


def load_forecasts(row: pd.Series, file_id: str, bundle, dataset: str) -> list[float]:
    output_type = str(row.get("t3_output_type", ""))
    if output_type == "LIST":
        return parse_forecast_list(str(row.get("t3_forecast_list", "")))
    if output_type == "SCRIPT":
        script_file = resolve_artifact_path(row.get("script_file", ""))
        if not script_file.exists():
            raise FileNotFoundError(f"script file not found: {script_file}")
        previous_log_path = EXECUTION_LOG_DIR / f"{file_id}.txt"
        if previous_log_path.exists() and not RERUN_FAILED_SCRIPTS:
            raise RuntimeError(f"previous script execution failed; see {previous_log_path}")
        # Turn-3 prompts now require scripts to read input.csv from the current
        # working directory. Scripts are executed sequentially, so refreshing the
        # shared file immediately before each run keeps old and new scripts usable.
        bundle.write_input_csv(script_file.parent / "input.csv")
        return execute_script(script_file, file_id, script_timeout_sec(dataset))
    raise ValueError(f"unsupported output type: {output_type}")


def load_existing_forecast_output(path: Path) -> list[float]:
    data = pd.read_csv(path)
    if "forecast" not in data.columns:
        raise ValueError(f"existing forecast output missing forecast column: {path}")
    return pd.to_numeric(data["forecast"], errors="raise").astype(float).tolist()


def script_timeout_sec(dataset: str) -> int:
    return DATASET_SCRIPT_TIMEOUT_SEC.get(dataset, SCRIPT_TIMEOUT_SEC)


def execute_script(script_file: Path, file_id: str, timeout_sec: int) -> list[float]:
    EXECUTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXECUTION_LOG_DIR / f"{file_id}.txt"
    script_file = prepare_script_for_execution(script_file)
    try:
        completed = subprocess.run(
            [sys.executable, script_file.name],
            cwd=script_file.parent,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log_path.write_text(
            "STDOUT\n"
            f"{exc.stdout or ''}\n\n"
            "STDERR\n"
            f"{exc.stderr or ''}\n\n"
            f"TIMEOUT after {timeout_sec} seconds",
            encoding="utf-8",
        )
        raise RuntimeError(f"script timed out after {timeout_sec}s; see {log_path}") from exc

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


def prepare_script_for_execution(script_file: Path) -> Path:
    """
    Clean harmless export artifacts before running generated scripts.

    Some parsed responses leave a closing markdown fence at EOF.  Rewriting the
    script keeps old parser output executable without requiring a full reparse.
    """
    content = script_file.read_text(encoding="utf-8")
    cleaned = strip_trailing_markdown_fences(content)
    if cleaned != content.rstrip():
        script_file.write_text(cleaned + "\n", encoding="utf-8")
    return script_file


def strip_trailing_markdown_fences(script: str) -> str:
    lines = script.rstrip().splitlines()
    while lines and re.fullmatch(r"\s*```\s*", lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def parse_forecast_list(text: str) -> list[float]:
    candidate = extract_last_list_literal(text)
    if not candidate:
        raise ValueError("no Python forecast list found")
    candidate = normalize_forecast_literal(candidate)
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
            ast.literal_eval(normalize_forecast_literal(candidate))
            return candidate
        except Exception:
            start = text.rfind("[", 0, start)
    return ""


def normalize_forecast_literal(text: str) -> str:
    """
    Convert common numpy scalar reprs in printed lists to plain literals.

    Generated scripts often print lists like [np.float64(1.2), ...].  Those are
    valid stdout for humans but not safe for ast.literal_eval until normalized.
    """
    scalar_call = re.compile(
        r"\b(?:np|numpy)\."
        r"(?:float(?:16|32|64)?|int(?:8|16|32|64)?|"
        r"uint(?:8|16|32|64)?|longdouble|float_)\(([^()]+)\)"
    )
    previous = None
    normalized = text
    while normalized != previous:
        previous = normalized
        normalized = scalar_call.sub(r"\1", normalized)
    return normalized


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
    out["actual"] = bundle.test_target_values()
    out["forecast"] = forecasts
    out.to_csv(path, index=False)


def branch_to_track(branch: str) -> str:
    return BRANCH_TO_TRACK.get(branch, "baseline")


def resolve_artifact_path(value: Any) -> Path:
    """
    Resolve artifact paths written by both old and current parser versions.

    Old parsed rows stored `llm-ts-protocolbench/results/...`; newer rows store
    `results/...`.  Both should work when metrics are recomputed later.
    """
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return BASE_DIR / "__missing_artifact__"

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate

    if raw.startswith("llm-ts-protocolbench/"):
        raw = raw.split("/", 1)[1]
    return BASE_DIR / raw


def to_repo_relative_path(path: Path) -> str:
    return str(path.relative_to(BASE_DIR)).replace("\\", "/")


def add_diebold_mariano_results(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare each successful challenger forecast with its paired Base forecast.

    Pairing key:
        dataset + scenario + llm_version + run_id

    DM loss:
        squared error, challenger minus Base.  Negative mean differential means
        the challenger forecast has lower average squared loss than Base.
    """
    out = metrics_df.copy()
    ok = out[out["execution_status"] == "OK"].copy()
    dm_rows: list[dict[str, Any]] = []
    forecast_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    horizon_cache: dict[tuple[str, str], int] = {}

    base_rows = ok[ok["branch"] == "Base"].copy()
    base_lookup = {
        dm_pair_key(row): row
        for _, row in base_rows.iterrows()
    }

    for idx, row in ok.iterrows():
        if row["branch"] == "Base":
            continue

        base_row = base_lookup.get(dm_pair_key(row))
        if base_row is None:
            continue

        actual, candidate = load_forecast_pair(row, forecast_cache)
        base_actual, base_forecast = load_forecast_pair(base_row, forecast_cache)
        if len(actual) != len(base_actual) or not np.allclose(actual, base_actual, equal_nan=True):
            logger.warning(
                "Skip DM for %s: actual series differs from base %s",
                row["file_id"],
                base_row["file_id"],
            )
            continue

        horizon_key = (str(row["dataset"]), str(row["scenario"]))
        if horizon_key not in horizon_cache:
            setup = load_model_setup(DEFAULT_MODEL_SETUP_PATH, *horizon_key, "baseline")
            horizon_cache[horizon_key] = max(int(setup.horizon_or_block) - 1, 0)
        lag = min(horizon_cache[horizon_key], max(len(actual) - 1, 0))
        dm_result = diebold_mariano_test(actual, candidate, base_forecast, lag=lag)

        out.loc[idx, "dm_reference_file_id"] = base_row["file_id"]
        out.loc[idx, "dm_reference_branch"] = "Base"
        out.loc[idx, "dm_loss"] = "squared_error"
        out.loc[idx, "dm_lag"] = lag
        out.loc[idx, "dm_n"] = dm_result["n"]
        out.loc[idx, "dm_mean_loss_diff"] = dm_result["mean_loss_diff"]
        out.loc[idx, "dm_statistic"] = dm_result["statistic"]
        out.loc[idx, "dm_p_value"] = dm_result["p_value"]
        out.loc[idx, "dm_better_than_base"] = bool(
            np.isfinite(dm_result["p_value"])
            and dm_result["p_value"] < 0.05
            and dm_result["mean_loss_diff"] < 0
        )

        dm_rows.append({
            "file_id": row["file_id"],
            "reference_file_id": base_row["file_id"],
            "dataset": row["dataset"],
            "scenario": row["scenario"],
            "llm_version": row["llm_version"],
            "run_id": row["run_id"],
            "branch": row["branch"],
            "reference_branch": "Base",
            "loss": "squared_error",
            "lag": lag,
            **dm_result,
            "challenger_better_than_base": bool(
                np.isfinite(dm_result["p_value"])
                and dm_result["p_value"] < 0.05
                and dm_result["mean_loss_diff"] < 0
            ),
        })

    dm_columns = [
        "file_id",
        "reference_file_id",
        "dataset",
        "scenario",
        "llm_version",
        "run_id",
        "branch",
        "reference_branch",
        "loss",
        "lag",
        "n",
        "mean_loss_diff",
        "statistic",
        "p_value",
        "challenger_better_than_base",
    ]
    return out, pd.DataFrame(dm_rows, columns=dm_columns)


def dm_pair_key(row: pd.Series) -> tuple[str, str, str, str]:
    return (
        str(row["dataset"]),
        str(row["scenario"]),
        str(row["llm_version"]),
        str(row["run_id"]),
    )


def load_forecast_pair(
    row: pd.Series,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    file_id = str(row["file_id"])
    if file_id not in cache:
        path = resolve_artifact_path(row["forecast_output_file"])
        data = pd.read_csv(path)
        cache[file_id] = (
            pd.to_numeric(data["actual"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(data["forecast"], errors="coerce").to_numpy(dtype=float),
        )
    return cache[file_id]


def diebold_mariano_test(
    actual: np.ndarray,
    candidate_forecast: np.ndarray,
    reference_forecast: np.ndarray,
    lag: int = 0,
) -> dict[str, float]:
    """
    Two-sided Diebold-Mariano test using squared-error loss differential.

    d_t = loss(candidate)_t - loss(reference)_t
    Negative mean_loss_diff indicates the candidate has lower average loss.
    """
    actual = np.asarray(actual, dtype=float)
    candidate_forecast = np.asarray(candidate_forecast, dtype=float)
    reference_forecast = np.asarray(reference_forecast, dtype=float)
    mask = (
        np.isfinite(actual)
        & np.isfinite(candidate_forecast)
        & np.isfinite(reference_forecast)
    )
    actual = actual[mask]
    candidate_forecast = candidate_forecast[mask]
    reference_forecast = reference_forecast[mask]
    n = len(actual)
    if n < 2:
        return {"n": n, "mean_loss_diff": np.nan, "statistic": np.nan, "p_value": np.nan}

    candidate_loss = (actual - candidate_forecast) ** 2
    reference_loss = (actual - reference_forecast) ** 2
    diff = candidate_loss - reference_loss
    mean_diff = float(np.mean(diff))
    lag = min(max(int(lag), 0), n - 1)
    long_run_var = newey_west_variance(diff, lag)
    if not np.isfinite(long_run_var) or long_run_var <= 0:
        return {"n": n, "mean_loss_diff": mean_diff, "statistic": np.nan, "p_value": np.nan}

    statistic = float(mean_diff / np.sqrt(long_run_var / n))
    p_value = float(2 * norm.sf(abs(statistic)))
    return {
        "n": n,
        "mean_loss_diff": mean_diff,
        "statistic": statistic,
        "p_value": p_value,
    }


def newey_west_variance(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    n = len(centered)
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        gamma_k = float(np.dot(centered[k:], centered[:-k]) / n)
        variance += 2 * weight * gamma_k
    return variance


if __name__ == "__main__":
    main()
