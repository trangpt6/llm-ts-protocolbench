"""Compare successful Part 2 runs against Part 0 traditional baselines."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from part2_3_compute_metrics import diebold_mariano_test, resolve_artifact_path
from part2_api.logging_utils import setup_logger
from part2_api.model_setup import load_model_setup
from part2_api.paths import DEFAULT_MODEL_SETUP_PATH, LOG_DIR, RESULTS_DIR


BASE_DIR = Path(__file__).resolve().parent.parent
PART0_RESULTS_PATH = BASE_DIR / "results" / "part0-traditional-baselines" / "baseline-traditional-results.csv"
PART2_RESULTS_DIR = RESULTS_DIR / "part2-interactive-llm-forecasting"
PART2_METRICS_PATH = PART2_RESULTS_DIR / "metrics-part2-interactive-llm-forecasting.csv"
SUMMARY_DIR = PART2_RESULTS_DIR / "summary-tables"
OUTPUT_PATH = SUMMARY_DIR / "part2-vs-part0-comparison.csv"
OUTPUT_BY_LLM_PATH = SUMMARY_DIR / "part2-vs-part0-comparison-by-llm.csv"
WIN_TIE_LOSS_OUTPUT_PATH = SUMMARY_DIR / "part2-vs-part0-win-tie-loss.csv"
DM_WIN_TIE_LOSS_OUTPUT_PATH = SUMMARY_DIR / "part2-vs-part0-dm-win-tie-loss.csv"
DM_OUTPUT_PATH = SUMMARY_DIR / "part2-vs-part0-diebold-mariano.csv"
METRICS = ["mae", "rmse", "smape", "mase", "r2"]

logger = logging.getLogger("part2_compare_with_part0")


def main() -> None:
    global logger
    logger = setup_logger("part2-compare-with-part0", LOG_DIR / "master-logs")
    if not PART0_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Part 0 results not found: {PART0_RESULTS_PATH}")
    if not PART2_METRICS_PATH.exists():
        raise FileNotFoundError(f"Part 2 metrics not found: {PART2_METRICS_PATH}")

    part0 = pd.read_csv(PART0_RESULTS_PATH)
    part2 = pd.read_csv(PART2_METRICS_PATH)
    ok = part2[part2["execution_status"] == "OK"].copy()
    if ok.empty:
        raise ValueError("No successful Part 2 metric rows found.")

    comparison = build_comparison(ok, part0)
    comparison_by_llm = build_comparison(ok, part0, extra_group_cols=["llm_version"])
    dm = build_dm_comparison(ok, part0)
    win_tie_loss = build_win_tie_loss_summary(comparison)
    dm_win_tie_loss = build_dm_win_tie_loss_summary(dm)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    comparison_by_llm.to_csv(OUTPUT_BY_LLM_PATH, index=False, encoding="utf-8-sig")
    win_tie_loss.to_csv(WIN_TIE_LOSS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    dm_win_tie_loss.to_csv(DM_WIN_TIE_LOSS_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    dm.to_csv(DM_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    logger.info("Saved Part 2 vs Part 0 comparison: %s rows=%d", OUTPUT_PATH, len(comparison))
    logger.info(
        "Saved Part 2 vs Part 0 comparison by LLM: %s rows=%d",
        OUTPUT_BY_LLM_PATH,
        len(comparison_by_llm),
    )
    logger.info(
        "Saved Part 2 vs Part 0 win/tie/loss summary: %s rows=%d",
        WIN_TIE_LOSS_OUTPUT_PATH,
        len(win_tie_loss),
    )
    logger.info(
        "Saved Part 2 vs Part 0 DM win/tie/loss summary: %s rows=%d",
        DM_WIN_TIE_LOSS_OUTPUT_PATH,
        len(dm_win_tie_loss),
    )
    logger.info("Saved Part 2 vs Part 0 DM comparison: %s rows=%d", DM_OUTPUT_PATH, len(dm))
    print(f"Saved comparison -> {OUTPUT_PATH}")
    print(f"Saved comparison by LLM -> {OUTPUT_BY_LLM_PATH}")
    print(f"Saved win/tie/loss summary -> {WIN_TIE_LOSS_OUTPUT_PATH}")
    print(f"Saved DM win/tie/loss summary -> {DM_WIN_TIE_LOSS_OUTPUT_PATH}")
    print(f"Saved DM comparison -> {DM_OUTPUT_PATH}")


def build_comparison(
    part2_ok: pd.DataFrame,
    part0: pd.DataFrame,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a long-form summary comparison over overlapping dataset/scenario pairs."""
    extra_group_cols = extra_group_cols or []
    group_cols = ["dataset", "scenario", "branch", *extra_group_cols]
    part2_summary = (
        part2_ok
        .groupby(group_cols, dropna=False)[METRICS]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    part2_summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in part2_summary.columns
    ]

    long_rows = []
    for _, row in part2_summary.iterrows():
        matches = part0[
            (part0["dataset"] == row["dataset"])
            & (part0["scenario"] == row["scenario"])
        ]
        for _, baseline in matches.iterrows():
            for metric in METRICS:
                part2_mean = row[f"{metric}_mean"]
                part0_value = baseline[metric]
                delta = part2_mean - part0_value
                part2_better = delta > 0 if metric == "r2" else delta < 0
                long_rows.append({
                    "dataset": row["dataset"],
                    "scenario": row["scenario"],
                    "branch": row["branch"],
                    **{col: row[col] for col in extra_group_cols},
                    "part0_model": baseline["model"],
                    "metric": metric,
                    "part2_n_runs": row[f"{metric}_count"],
                    "part2_mean": part2_mean,
                    "part2_std": row[f"{metric}_std"],
                    "part0_value": part0_value,
                    "delta_part2_minus_part0": delta,
                    "part2_better": bool(part2_better),
                })

    return pd.DataFrame(long_rows)


def build_win_tie_loss_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compact performance win/tie/loss table from mean metric deltas."""
    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "branch",
                "part0_model",
                "metric",
                "comparisons",
                "wins",
                "ties",
                "losses",
                "win_rate_pct",
                "mean_delta_part2_minus_part0",
            ]
        )

    frame = comparison.copy()
    frame["delta_part2_minus_part0"] = pd.to_numeric(
        frame["delta_part2_minus_part0"],
        errors="coerce",
    )
    frame = frame.dropna(subset=["delta_part2_minus_part0"])
    frame["outcome"] = np.select(
        [
            np.isclose(frame["delta_part2_minus_part0"], 0.0, rtol=1e-9, atol=1e-12),
            frame["part2_better"].astype(bool),
        ],
        ["tie", "win"],
        default="loss",
    )
    counts = (
        frame.groupby(["branch", "part0_model", "metric", "outcome"], dropna=False)
        .size()
        .unstack("outcome", fill_value=0)
        .reset_index()
    )
    for column in ["win", "tie", "loss"]:
        if column not in counts.columns:
            counts[column] = 0
    means = (
        frame.groupby(["branch", "part0_model", "metric"], dropna=False)["delta_part2_minus_part0"]
        .mean()
        .reset_index(name="mean_delta_part2_minus_part0")
    )
    summary = counts.merge(means, on=["branch", "part0_model", "metric"], how="left")
    summary = summary.rename(columns={"win": "wins", "tie": "ties", "loss": "losses"})
    summary["comparisons"] = summary["wins"] + summary["ties"] + summary["losses"]
    summary["win_rate_pct"] = np.where(
        summary["comparisons"] > 0,
        (summary["wins"] / summary["comparisons"] * 100).round(2),
        0.0,
    )
    columns = [
        "branch",
        "part0_model",
        "metric",
        "comparisons",
        "wins",
        "ties",
        "losses",
        "win_rate_pct",
        "mean_delta_part2_minus_part0",
    ]
    return summary[columns].sort_values(["metric", "branch", "part0_model"]).reset_index(drop=True)


def build_dm_win_tie_loss_summary(dm: pd.DataFrame) -> pd.DataFrame:
    """Compact significant win/loss counts from DM tests."""
    columns = [
        "branch",
        "part0_model",
        "dm_tests",
        "significant_wins",
        "ties_or_not_significant",
        "significant_losses",
        "significant_win_rate_pct",
        "mean_loss_diff",
    ]
    if dm.empty:
        return pd.DataFrame(columns=columns)

    frame = dm.copy()
    frame["mean_loss_diff"] = pd.to_numeric(frame["mean_loss_diff"], errors="coerce")
    for column in ["significant_win", "significant_loss"]:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].fillna(False).astype(bool)

    grouped = frame.groupby(["branch", "part0_model"], dropna=False)
    summary = grouped.agg(
        dm_tests=("file_id", "size"),
        significant_wins=("significant_win", "sum"),
        significant_losses=("significant_loss", "sum"),
        mean_loss_diff=("mean_loss_diff", "mean"),
    ).reset_index()
    summary["ties_or_not_significant"] = (
        summary["dm_tests"] - summary["significant_wins"] - summary["significant_losses"]
    )
    summary["significant_win_rate_pct"] = np.where(
        summary["dm_tests"] > 0,
        (summary["significant_wins"] / summary["dm_tests"] * 100).round(2),
        0.0,
    )
    return summary[columns].sort_values(["branch", "part0_model"]).reset_index(drop=True)


def build_dm_comparison(part2_ok: pd.DataFrame, part0: pd.DataFrame) -> pd.DataFrame:
    """Run DM tests for every Part 2 forecast paired with each matching Part 0 forecast."""
    rows: list[dict[str, Any]] = []
    part2_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    part0_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    lag_cache: dict[tuple[str, str], int] = {}

    columns = [
        "file_id",
        "dataset",
        "scenario",
        "branch",
        "llm_version",
        "run_id",
        "part0_model",
        "loss",
        "lag",
        "n",
        "mean_loss_diff",
        "statistic",
        "p_value",
        "significant_win",
        "significant_loss",
        "part2_better_than_part0",
    ]
    if "forecast_output_file" not in part0.columns or "forecast_output_file" not in part2_ok.columns:
        return pd.DataFrame(columns=columns)

    usable_part0 = part0[part0["forecast_output_file"].fillna("").astype(str).str.len() > 0].copy()
    usable_part2 = part2_ok[part2_ok["forecast_output_file"].fillna("").astype(str).str.len() > 0].copy()

    for _, candidate in usable_part2.iterrows():
        matches = usable_part0[
            (usable_part0["dataset"] == candidate["dataset"])
            & (usable_part0["scenario"] == candidate["scenario"])
        ]
        for _, baseline in matches.iterrows():
            actual, candidate_forecast = load_forecast_pair(
                candidate["forecast_output_file"],
                part2_cache,
                resolve_artifact_path,
            )
            base_actual, base_forecast = load_forecast_pair(
                baseline["forecast_output_file"],
                part0_cache,
                resolve_part0_path,
            )
            if len(actual) != len(base_actual) or not np.allclose(actual, base_actual, equal_nan=True):
                logger.warning(
                    "Skip Part 2 vs Part 0 DM for %s vs %s: actual series differs",
                    candidate.get("file_id", ""),
                    baseline["model"],
                )
                continue

            horizon_key = (str(candidate["dataset"]), str(candidate["scenario"]))
            if horizon_key not in lag_cache:
                setup = load_model_setup(DEFAULT_MODEL_SETUP_PATH, *horizon_key, "baseline")
                lag_cache[horizon_key] = max(int(setup.horizon_or_block) - 1, 0)
            lag = min(lag_cache[horizon_key], max(len(actual) - 1, 0))
            dm = diebold_mariano_test(actual, candidate_forecast, base_forecast, lag=lag)
            significant_win = bool(
                np.isfinite(dm["p_value"])
                and dm["p_value"] < 0.05
                and dm["mean_loss_diff"] < 0
            )
            significant_loss = bool(
                np.isfinite(dm["p_value"])
                and dm["p_value"] < 0.05
                and dm["mean_loss_diff"] > 0
            )
            rows.append({
                "file_id": candidate.get("file_id", ""),
                "dataset": candidate["dataset"],
                "scenario": candidate["scenario"],
                "branch": candidate["branch"],
                "llm_version": candidate.get("llm_version", ""),
                "run_id": candidate.get("run_id", ""),
                "part0_model": baseline["model"],
                "loss": "squared_error",
                "lag": lag,
                **dm,
                "significant_win": significant_win,
                "significant_loss": significant_loss,
                "part2_better_than_part0": significant_win,
            })

    return pd.DataFrame(rows, columns=columns)


def load_forecast_pair(
    path_value: object,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
    resolver,
) -> tuple[np.ndarray, np.ndarray]:
    key = str(path_value)
    if key not in cache:
        data = pd.read_csv(resolver(path_value))
        cache[key] = (
            pd.to_numeric(data["actual"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(data["forecast"], errors="coerce").to_numpy(dtype=float),
        )
    return cache[key]


def resolve_part0_path(value: object) -> Path:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return BASE_DIR / "__missing_part0_forecast__"
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / raw


if __name__ == "__main__":
    main()
