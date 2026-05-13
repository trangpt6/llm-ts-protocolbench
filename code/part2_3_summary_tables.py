"""
Build Part 2 summary tables from per-run metrics.

Input:
    results/part2-interactive-llm-forecasting/part2-metrics.csv

Outputs:
    results/part2-interactive-llm-forecasting/summary-tables/*.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from part2_api.logging_utils import setup_logger
from part2_api.paths import LOG_DIR, RESULTS_DIR


PART2_RESULTS_DIR = RESULTS_DIR / "part2-interactive-llm-forecasting"
METRICS_PATH = PART2_RESULTS_DIR / "metrics-part2-interactive-llm-forecasting.csv"
SUMMARY_DIR = PART2_RESULTS_DIR / "summary-tables"
METRICS = ["mae", "rmse", "smape", "mase", "r2"]

logger = logging.getLogger("part2_summary_tables")


def main() -> None:
    global logger
    logger = setup_logger("part2_summary_tables", LOG_DIR / "master-logs")
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")
    df = pd.read_csv(METRICS_PATH)
    ok = df[df["execution_status"] == "OK"].copy()
    if ok.empty:
        raise ValueError("No successful metric rows found.")

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    write_summary(ok, ["llm_version"], "part2-summary-by-llm.csv")
    write_summary(ok, ["llm_version", "branch"], "part2-summary-by-llm-branch.csv")
    write_summary(ok, ["dataset", "scenario"], "part2-summary-by-dataset-scenario.csv")
    write_summary(ok, ["scenario"], "part2-summary-by-scenario.csv")
    logger.info("Saved summary tables to %s", SUMMARY_DIR)
    print(f"Saved summary tables -> {SUMMARY_DIR}")


def write_summary(df: pd.DataFrame, group_cols: list[str], filename: str) -> None:
    grouped = df.groupby(group_cols, dropna=False)[METRICS].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    grouped.to_csv(SUMMARY_DIR / filename, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
