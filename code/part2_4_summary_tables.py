"""Build Part 2 summary tables, coverage checks, and paired comparisons."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from part2_api.logging_utils import setup_logger
from part2_api.paths import LOG_DIR, RESULTS_DIR


PART2_RESULTS_DIR = RESULTS_DIR / "part2-interactive-llm-forecasting"
METRICS_PATH = PART2_RESULTS_DIR / "metrics-part2-interactive-llm-forecasting.csv"
SUMMARY_DIR = PART2_RESULTS_DIR / "summary-tables"
METRICS = ["mae", "rmse", "smape", "mase", "r2"]
PAIR_COLS = ["dataset", "scenario", "llm_version", "run_id"]
BRANCHES = ["Base", "ChalML", "ChalDL"]
CHALLENGER_BRANCHES = ["ChalML", "ChalDL"]

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
    build_run_coverage(df).to_csv(
        SUMMARY_DIR / "part2-run-coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_summary(ok, ["llm_version"], "part2-summary-by-llm.csv")
    write_summary(ok, ["llm_version", "branch"], "part2-summary-by-llm-branch.csv")
    write_summary(ok, ["dataset", "scenario"], "part2-summary-by-dataset-scenario.csv")
    write_summary(ok, ["scenario"], "part2-summary-by-scenario.csv")
    write_paired_metric_summary(ok, [], "part2-paired-summary-vs-base-overall.csv")
    write_paired_metric_summary(ok, ["llm_version"], "part2-paired-summary-vs-base-by-llm.csv")
    write_paired_metric_summary(
        ok,
        ["dataset", "scenario"],
        "part2-paired-summary-vs-base-by-dataset-scenario.csv",
    )
    write_wilcoxon_vs_base(ok, "part2-wilcoxon-vs-base.csv")
    logger.info("Saved summary tables to %s", SUMMARY_DIR)
    print(f"Saved summary tables -> {SUMMARY_DIR}")


def write_summary(df: pd.DataFrame, group_cols: list[str], filename: str) -> None:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        for metric in METRICS:
            row.update(metric_stats(group[metric], f"{metric}_"))
        rows.append(row)
    pd.DataFrame(rows).to_csv(SUMMARY_DIR / filename, index=False, encoding="utf-8-sig")


def metric_stats(values: pd.Series | np.ndarray, prefix: str) -> dict[str, float | int]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(numeric)
    if n == 0:
        return {
            f"{prefix}count": 0,
            f"{prefix}mean": np.nan,
            f"{prefix}std": np.nan,
            f"{prefix}median": np.nan,
            f"{prefix}q1": np.nan,
            f"{prefix}q3": np.nan,
            f"{prefix}iqr": np.nan,
            f"{prefix}ci95_low": np.nan,
            f"{prefix}ci95_high": np.nan,
        }

    mean = float(np.mean(numeric))
    std = float(np.std(numeric, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    q1 = float(np.percentile(numeric, 25))
    q3 = float(np.percentile(numeric, 75))
    return {
        f"{prefix}count": n,
        f"{prefix}mean": mean,
        f"{prefix}std": std,
        f"{prefix}median": float(np.median(numeric)),
        f"{prefix}q1": q1,
        f"{prefix}q3": q3,
        f"{prefix}iqr": q3 - q1,
        f"{prefix}ci95_low": mean - 1.96 * sem,
        f"{prefix}ci95_high": mean + 1.96 * sem,
    }


def build_run_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize available and paired successful runs.

    This table is intentionally based on all metric rows, not only successful
    rows, so partial ETTh1 batches and failed outputs are visible before any
    performance interpretation.
    """
    rows = []
    group_cols = ["dataset", "scenario", "llm_version"]
    for keys, group in df.groupby(group_cols, dropna=False):
        dataset, scenario, llm_version = keys
        ok_group = group[group["execution_status"] == "OK"].copy()
        ok_run_ids = {
            branch: set(ok_group.loc[ok_group["branch"] == branch, "run_id"].astype(str))
            for branch in BRANCHES
        }
        base_ids = ok_run_ids["Base"]
        complete_ids = set.intersection(*(ok_run_ids[branch] for branch in BRANCHES))
        row = {
            "dataset": dataset,
            "scenario": scenario,
            "llm_version": llm_version,
            "total_rows": len(group),
            "ok_rows": len(ok_group),
            "failed_rows": int((group["execution_status"] != "OK").sum()),
            "complete_three_branch_runs": len(complete_ids),
        }
        for branch in BRANCHES:
            branch_group = group[group["branch"] == branch]
            branch_ok = ok_group[ok_group["branch"] == branch]
            row[f"{branch}_rows"] = len(branch_group)
            row[f"{branch}_ok_rows"] = len(branch_ok)
            row[f"{branch}_ok_run_ids"] = ",".join(sorted(ok_run_ids[branch]))
        for branch in CHALLENGER_BRANCHES:
            row[f"{branch}_paired_with_base_runs"] = len(base_ids & ok_run_ids[branch])
        rows.append(row)

    columns = [
        "dataset",
        "scenario",
        "llm_version",
        "total_rows",
        "ok_rows",
        "failed_rows",
        "Base_rows",
        "Base_ok_rows",
        "Base_ok_run_ids",
        "ChalML_rows",
        "ChalML_ok_rows",
        "ChalML_ok_run_ids",
        "ChalDL_rows",
        "ChalDL_ok_rows",
        "ChalDL_ok_run_ids",
        "ChalML_paired_with_base_runs",
        "ChalDL_paired_with_base_runs",
        "complete_three_branch_runs",
    ]
    return pd.DataFrame(rows).reindex(columns=columns)


def write_paired_metric_summary(df: pd.DataFrame, group_cols: list[str], filename: str) -> None:
    """
    Summarize only challenger runs that have a matched Base run.

    Pairing key:
        dataset + scenario + llm_version + run_id

    These paired summaries are the safe tables for Base-vs-challenger
    interpretation when the experiment has partial or unbalanced LLM batches.
    """
    rows = []
    for branch in CHALLENGER_BRANCHES:
        base = df[df["branch"] == "Base"][PAIR_COLS + METRICS].copy()
        challenger = df[df["branch"] == branch][PAIR_COLS + METRICS].copy()
        paired = base.merge(
            challenger,
            on=PAIR_COLS,
            suffixes=("_base", "_challenger"),
            how="inner",
        )
        if paired.empty:
            continue

        grouped = (
            [((), paired)]
            if not group_cols
            else paired.groupby(group_cols, dropna=False)
        )
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            prefix = {col: value for col, value in zip(group_cols, keys)}
            row = {
                **prefix,
                "challenger_branch": branch,
                "n_pairs": len(group),
            }
            for metric in METRICS:
                base_values = pd.to_numeric(group[f"{metric}_base"], errors="coerce")
                challenger_values = pd.to_numeric(group[f"{metric}_challenger"], errors="coerce")
                valid = base_values.notna() & challenger_values.notna()
                base_metric = base_values[valid].to_numpy(dtype=float)
                challenger_metric = challenger_values[valid].to_numpy(dtype=float)
                differences = challenger_metric - base_metric
                row.update(metric_stats(base_metric, f"{metric}_base_"))
                row.update(metric_stats(challenger_metric, f"{metric}_challenger_"))
                row.update(metric_stats(differences, f"{metric}_difference_"))
                row[f"{metric}_mean_difference"] = row[f"{metric}_difference_mean"]
                row[f"{metric}_median_difference"] = row[f"{metric}_difference_median"]
                row[f"{metric}_challenger_better_by_mean"] = bool(
                    len(differences)
                    and (
                        np.mean(differences) < 0
                        if metric != "r2"
                        else np.mean(differences) > 0
                    )
                )
            rows.append(row)

    pd.DataFrame(rows).to_csv(SUMMARY_DIR / filename, index=False, encoding="utf-8-sig")


def write_wilcoxon_vs_base(df: pd.DataFrame, filename: str) -> None:
    """
    Paired Wilcoxon signed-rank tests for final aggregate comparisons.

    Pairing key:
        dataset + scenario + llm_version + run_id

    Each challenger branch is compared against Base across all matched runs.
    """
    rows = []
    for branch in CHALLENGER_BRANCHES:
        base = df[df["branch"] == "Base"][PAIR_COLS + METRICS].copy()
        challenger = df[df["branch"] == branch][PAIR_COLS + METRICS].copy()
        paired = base.merge(
            challenger,
            on=PAIR_COLS,
            suffixes=("_base", "_challenger"),
            how="inner",
        )

        for metric in METRICS:
            base_values = pd.to_numeric(paired[f"{metric}_base"], errors="coerce")
            challenger_values = pd.to_numeric(paired[f"{metric}_challenger"], errors="coerce")
            valid = base_values.notna() & challenger_values.notna()
            base_metric = base_values[valid].to_numpy(dtype=float)
            challenger_metric = challenger_values[valid].to_numpy(dtype=float)
            differences = challenger_metric - base_metric
            nonzero = differences[np.isfinite(differences)]

            statistic = p_value = np.nan
            if len(nonzero) > 0:
                if np.allclose(nonzero, 0):
                    statistic = 0.0
                    p_value = 1.0
                else:
                    result = wilcoxon(
                        challenger_metric,
                        base_metric,
                        alternative="two-sided",
                        zero_method="wilcox",
                    )
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)

            rows.append({
                "branch": branch,
                "metric": metric,
                "n_pairs": len(base_metric),
                "base_median": float(np.median(base_metric)) if len(base_metric) else np.nan,
                "challenger_median": float(np.median(challenger_metric)) if len(challenger_metric) else np.nan,
                "median_difference": float(np.median(differences)) if len(differences) else np.nan,
                "statistic": statistic,
                "p_value": p_value,
                "challenger_better_by_median": bool(
                    len(differences)
                    and (
                        np.median(differences) < 0
                        if metric != "r2"
                        else np.median(differences) > 0
                    )
                ),
            })

    pd.DataFrame(rows).to_csv(SUMMARY_DIR / filename, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
