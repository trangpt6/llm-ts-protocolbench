"""Summarize Part 2 execution failures by LLM and dataset.

Run this after code/part2_3_compute_metrics.py has completed.
This script stays separate from metric computation and only summarizes
execution status, failure groups, and detailed failure categories.

Inputs:
    results/part2-interactive-llm-forecasting/metrics-part2-interactive-llm-forecasting.csv
    results/part2-interactive-llm-forecasting/execution-logs/<file_id>.txt

Outputs:
    results/part2-interactive-llm-forecasting/summary/part2-status-by-llm.csv
    results/part2-interactive-llm-forecasting/summary/part2-status-by-dataset.csv
    results/part2-interactive-llm-forecasting/summary/part2-status-by-llm-dataset.csv
    results/part2-interactive-llm-forecasting/summary/part2-error-groups-by-llm.csv
    results/part2-interactive-llm-forecasting/summary/part2-error-groups-by-dataset.csv
    results/part2-interactive-llm-forecasting/summary/part2-error-groups-by-llm-dataset.csv
    results/part2-interactive-llm-forecasting/summary/part2-error-categories-by-llm.csv
    results/part2-interactive-llm-forecasting/summary/part2-error-categories-by-dataset.csv
    results/part2-interactive-llm-forecasting/summary/part2-error-categories-by-llm-dataset.csv
    results/part2-interactive-llm-forecasting/summary/part2-failure-details.csv
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PART2_RESULTS_DIR = BASE_DIR / "results" / "part2-interactive-llm-forecasting"
METRICS_PATH = PART2_RESULTS_DIR / "metrics-part2-interactive-llm-forecasting.csv"
SUMMARY_DIR = PART2_RESULTS_DIR / "summary"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

    df = pd.read_csv(METRICS_PATH)
    df = add_error_categories(df)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    write_status_summary(df, ["llm_version"], "part2-status-by-llm.csv")
    write_status_summary(df, ["dataset"], "part2-status-by-dataset.csv")
    write_status_summary(df, ["llm_version", "dataset"], "part2-status-by-llm-dataset.csv")

    write_error_group_summary(df, ["llm_version"], "part2-error-groups-by-llm.csv")
    write_error_group_summary(df, ["dataset"], "part2-error-groups-by-dataset.csv")
    write_error_group_summary(
        df,
        ["llm_version", "dataset"],
        "part2-error-groups-by-llm-dataset.csv",
    )

    write_error_category_summary(df, ["llm_version"], "part2-error-categories-by-llm.csv")
    write_error_category_summary(df, ["dataset"], "part2-error-categories-by-dataset.csv")
    write_error_category_summary(
        df,
        ["llm_version", "dataset"],
        "part2-error-categories-by-llm-dataset.csv",
    )

    write_failure_details(df)
    print(f"Saved Part 2 execution failure summaries -> {SUMMARY_DIR}")


def add_error_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error"] = out.get("error", "").fillna("").astype(str)
    out["output_type"] = out.get("output_type", "").fillna("").astype(str)
    out["execution_status"] = out.get("execution_status", "").fillna("").astype(str)
    out["execution_log_tail"] = out.apply(read_execution_log_tail, axis=1)
    out["error_category"] = out.apply(classify_error, axis=1)

    # ---- Deep-log analysis for OTHER_FAIL entries ----
    # OTHER_FAIL means "previous script execution failed" — the real error
    # is buried in the execution log.  Read full logs and re-classify.
    other_mask = out["error_category"] == "OTHER_FAIL"
    if other_mask.any():
        for idx in out[other_mask].index:
            row = out.loc[idx]
            path_text = str(row.get("execution_log_file", "") or "").strip()
            if not path_text:
                continue
            log_path = resolve_repo_path(path_text)
            if not log_path.exists():
                continue
            try:
                content = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            new_cat = classify_exec_log_deep(content)
            if new_cat and new_cat != "OTHER_FAIL":
                out.at[idx, "error_category"] = new_cat

    out["error_group"] = out["error_category"].map(error_group).fillna("runtime/infra issue")
    return out


def classify_error(row: pd.Series) -> str:
    status = str(row.get("execution_status", "")).upper()
    output_type = str(row.get("output_type", "")).upper()
    error = str(row.get("error", ""))
    log_tail = str(row.get("execution_log_tail", ""))
    combined = f"{error}\n{log_tail}"
    combined_lower = combined.lower()

    if status == "OK":
        return "OK"
    if output_type == "ERROR" or "unsupported output type: error" in combined_lower:
        return "MODEL_RETURNED_ERROR"
    if output_type == "TEXT" or "unsupported output type: text" in combined_lower:
        return "UNSTRUCTURED_TEXT_OUTPUT"
    if "timed out" in combined_lower or "timeout after" in combined_lower:
        return "TIMEOUT"
    if "forecast length mismatch" in combined_lower:
        return "FORECAST_LENGTH_MISMATCH"
    if "no python forecast list found" in combined_lower:
        return "NO_FORECAST_LIST_FOUND"
    if "syntaxerror" in combined_lower or "indentationerror" in combined_lower:
        return "SCRIPT_SYNTAX_ERROR"
    if "modulenotfounderror" in combined_lower or "importerror" in combined_lower:
        return "MISSING_PACKAGE"
    if "script exited with code" in combined_lower:
        return classify_runtime_error(log_tail)
    if error.strip():
        return "OTHER_FAIL"
    return "FAIL_UNKNOWN"


def classify_runtime_error(log_tail: str) -> str:
    text = log_tail.lower()
    if "valueerror" in text:
        if "invalid frequency" in text:
            return "SCRIPT_PANDAS_FREQUENCY_ERROR"
        return "SCRIPT_VALUE_ERROR"
    if "keyerror" in text:
        return "SCRIPT_KEY_ERROR"
    if "indexerror" in text:
        return "SCRIPT_INDEX_ERROR"
    if "runtimeerror" in text and "tensor" in text:
        return "SCRIPT_TENSOR_SHAPE_ERROR"
    if "runtimeerror" in text:
        return "SCRIPT_RUNTIME_ERROR"
    if "typeerror" in text:
        return "SCRIPT_TYPE_ERROR"
    if "attributeerror" in text:
        return "SCRIPT_ATTRIBUTE_ERROR"
    if "nameerror" in text:
        return "SCRIPT_NAME_ERROR"
    return "SCRIPT_RUNTIME_ERROR"


def error_group(category: str) -> str:
    if category == "OK":
        return "OK"
    if category in {
        "MODEL_RETURNED_ERROR",
        "UNSTRUCTURED_TEXT_OUTPUT",
        "FORECAST_LENGTH_MISMATCH",
        "FORECAST_NUMPY_WRAPPED",
        "FORECAST_TENSOR_WRAPPED",
        "FORECAST_PARSE_FAILURE",
        "NO_FORECAST_LIST_FOUND",
    }:
        return "model/output issue"
    if category.startswith("SCRIPT_"):
        return "code issue"
    if category in {
        "TIMEOUT",
        "MISSING_PACKAGE",
        "OTHER_FAIL",
        "OTHER_FAIL_UNCLASSIFIED",
        "FAIL_UNKNOWN",
    }:
        return "runtime/infra issue"
    return "runtime/infra issue"


# ---------------------------------------------------------------------------
# Deep execution-log classifier for OTHER_FAIL decomposition.
# When the surface-level error is just "previous script execution failed",
# this function opens the full execution log and extracts the real error.
# ---------------------------------------------------------------------------
def classify_exec_log_deep(content: str) -> str:
    """Classify the actual error from a full execution log's content."""
    has_stdout = bool(re.search(r'(?:^|\n)STDOUT\s*\n\s*\S', content))
    has_stderr = bool(re.search(r'(?:^|\n)STDERR\s*\n\s*\S', content))
    full_lower = content.lower()

    # 1. Syntax / Indentation errors
    if "syntaxerror" in full_lower or "indentationerror" in full_lower:
        return "SCRIPT_SYNTAX_ERROR"

    # 2. Timeout (script timed out during execution)
    if "timed out" in full_lower or "timeout after" in full_lower:
        return "TIMEOUT"

    # 3. Missing imports
    if "nameerror" in full_lower:
        if "'pd' is not defined" in full_lower or "'np' is not defined" in full_lower:
            return "SCRIPT_MISSING_IMPORT"
        if "torch" in full_lower and "is not defined" in full_lower:
            return "SCRIPT_MISSING_IMPORT"
        return "SCRIPT_NAME_ERROR"

    # 4. AttributeError (typos like np.random.sed)
    if "attributeerror" in full_lower:
        return "SCRIPT_ATTRIBUTE_ERROR"

    # 5. ValueError (shape mismatch, array concatenation, etc.)
    if "valueerror" in full_lower:
        return "SCRIPT_VALUE_ERROR"

    # 6. TypeError (fillna method='ffill' in pandas 2.x, etc.)
    if "typeerror" in full_lower:
        return "SCRIPT_TYPE_ERROR"

    # 7. KeyError (column not found, etc.)
    if "keyerror" in full_lower:
        return "SCRIPT_KEY_ERROR"

    # 8. ImportError / ModuleNotFoundError
    if "importerror" in full_lower or "modulenotfounderror" in full_lower:
        return "SCRIPT_MISSING_PACKAGE"

    # 9. Framework-specific errors from traceback
    if "traceback" in full_lower:
        if "torch" in full_lower:
            return "SCRIPT_TORCH_ERROR"
        if "statsmodels" in full_lower:
            return "SCRIPT_STATSMODELS_ERROR"
        if "pandas" in full_lower:
            return "SCRIPT_PANDAS_ERROR"
        if "sklearn" in full_lower or "xgboost" in full_lower or "lightgbm" in full_lower:
            return "SCRIPT_SKLEARN_ERROR"
        return "SCRIPT_RUNTIME_ERROR"

    # 10. No STDERR but STDOUT has output → forecast parse issue
    if has_stdout and not has_stderr:
        stdout_content = content.split("STDOUT", 1)[1] if "STDOUT" in content else ""
        if "np.float" in stdout_content or "np.int" in stdout_content:
            return "FORECAST_NUMPY_WRAPPED"
        if "tensor(" in stdout_content:
            return "FORECAST_TENSOR_WRAPPED"
        if re.search(r'[\d.]+', stdout_content):
            return "FORECAST_PARSE_FAILURE"
        return "SCRIPT_EMPTY_OUTPUT"

    # 11. Only warnings (sklearn UserWarning, statsmodels ValueWarning) in output
    if "userwarning" in full_lower or "valuewarning" in full_lower:
        if "sklearn" in full_lower or "lgbmregressor" in full_lower or "lightgbm" in full_lower:
            return "SCRIPT_MODEL_WARNING_OUTPUT"
        if "statsmodels" in full_lower:
            return "SCRIPT_STATSMODELS_WARNING"
        if "torch" in full_lower:
            return "SCRIPT_TORCH_WARNING"
        return "FORECAST_PARSE_FAILURE"

    # 12. No STDOUT, no STDERR, but log exists
    if not has_stdout and not has_stderr:
        return "SCRIPT_NO_OUTPUT"

    return "OTHER_FAIL"  # genuinely unclassifiable — should be rare


def read_execution_log_tail(row: pd.Series, max_lines: int = 40) -> str:
    path_text = str(row.get("execution_log_file", "") or "").strip()
    if not path_text:
        return ""
    path = resolve_repo_path(path_text)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def resolve_repo_path(value: str) -> Path:
    raw = value.replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return path
    if raw.startswith("llm-ts-protocolbench/"):
        raw = raw.split("/", 1)[1]
    return BASE_DIR / raw


def write_status_summary(df: pd.DataFrame, group_cols: list[str], filename: str) -> None:
    grouped = df.groupby(group_cols, dropna=False)
    summary = grouped.apply(status_row).reset_index()
    summary = sort_summary(summary, group_cols)
    summary.to_csv(SUMMARY_DIR / filename, index=False, encoding="utf-8-sig")


def status_row(group: pd.DataFrame) -> pd.Series:
    total = len(group)
    ok = int((group["execution_status"] == "OK").sum())
    fail = total - ok
    script_rows = int((group["output_type"] == "SCRIPT").sum())
    list_rows = int((group["output_type"] == "LIST").sum())
    model_error_rows = int((group["output_type"] == "ERROR").sum())
    text_rows = int((group["output_type"] == "TEXT").sum())
    timeout = int((group["error_category"] == "TIMEOUT").sum())
    length_mismatch = int((group["error_category"] == "FORECAST_LENGTH_MISMATCH").sum())
    code_error = int(group["error_category"].str.startswith("SCRIPT_").sum())
    model_output_issue = int((group["error_group"] == "model/output issue").sum())
    code_issue = int((group["error_group"] == "code issue").sum())
    runtime_infra_issue = int((group["error_group"] == "runtime/infra issue").sum())
    return pd.Series({
        "total_runs": total,
        "ok_runs": ok,
        "fail_runs": fail,
        "ok_rate_pct": pct(ok, total),
        "fail_rate_pct": pct(fail, total),
        "script_rows": script_rows,
        "list_rows": list_rows,
        "model_error_rows": model_error_rows,
        "text_rows": text_rows,
        "timeout_runs": timeout,
        "timeout_rate_pct": pct(timeout, total),
        "length_mismatch_runs": length_mismatch,
        "length_mismatch_rate_pct": pct(length_mismatch, total),
        "code_error_runs": code_error,
        "code_error_rate_pct": pct(code_error, total),
        "model_output_issue_runs": model_output_issue,
        "model_output_issue_rate_pct": pct(model_output_issue, total),
        "code_issue_runs": code_issue,
        "code_issue_rate_pct": pct(code_issue, total),
        "runtime_infra_issue_runs": runtime_infra_issue,
        "runtime_infra_issue_rate_pct": pct(runtime_infra_issue, total),
    })


def write_error_group_summary(df: pd.DataFrame, group_cols: list[str], filename: str) -> None:
    counts = (
        df.groupby(group_cols + ["error_group"], dropna=False)
        .size()
        .reset_index(name="runs")
    )
    totals = df.groupby(group_cols, dropna=False).size().reset_index(name="total_runs")
    summary = counts.merge(totals, on=group_cols, how="left")
    summary["rate_pct"] = summary.apply(lambda row: pct(row["runs"], row["total_runs"]), axis=1)
    summary = summary.sort_values(group_cols + ["runs", "error_group"], ascending=[True] * len(group_cols) + [False, True])
    summary.to_csv(SUMMARY_DIR / filename, index=False, encoding="utf-8-sig")


def write_error_category_summary(df: pd.DataFrame, group_cols: list[str], filename: str) -> None:
    counts = (
        df.groupby(group_cols + ["error_category"], dropna=False)
        .size()
        .reset_index(name="runs")
    )
    totals = df.groupby(group_cols, dropna=False).size().reset_index(name="total_runs")
    summary = counts.merge(totals, on=group_cols, how="left")
    summary["rate_pct"] = summary.apply(lambda row: pct(row["runs"], row["total_runs"]), axis=1)
    summary = summary.sort_values(group_cols + ["runs", "error_category"], ascending=[True] * len(group_cols) + [False, True])
    summary.to_csv(SUMMARY_DIR / filename, index=False, encoding="utf-8-sig")


def write_failure_details(df: pd.DataFrame) -> None:
    columns = [
        "file_id",
        "dataset",
        "scenario",
        "branch",
        "llm_version",
        "run_id",
        "output_type",
        "execution_status",
        "forecast_length",
        "expected_length",
        "error_group",
        "error_category",
        "error",
        "execution_log_file",
    ]
    available = [col for col in columns if col in df.columns]
    failures = df[df["execution_status"] != "OK"].copy()
    failures = failures.sort_values(["dataset", "scenario", "branch", "llm_version", "run_id"])
    failures[available].to_csv(
        SUMMARY_DIR / "part2-failure-details.csv",
        index=False,
        encoding="utf-8-sig",
    )


def sort_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    sort_cols = group_cols + ["fail_rate_pct", "fail_runs"]
    ascending = [True] * len(group_cols) + [False, False]
    return df.sort_values(sort_cols, ascending=ascending)


def pct(numerator: Any, denominator: Any) -> float:
    denominator = int(denominator)
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator * 100, 2)


if __name__ == "__main__":
    main()
