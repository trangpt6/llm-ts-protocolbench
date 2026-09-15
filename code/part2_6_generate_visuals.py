"""Generate publication-ready tables and visualizations for the study.

This script is intentionally separate from the Part 0, Part 1, and Part 2
execution pipeline.  It only reads existing artifacts and writes visualization
assets under ``results/visualizations``.

Examples:
    python code/part2_6_generate_visuals.py --phase pre
    python code/part2_6_generate_visuals.py --phase post
    python code/part2_6_generate_visuals.py --phase all --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
PART1_SETUP_PATH = RESULTS_DIR / "part1-llm-strategic-consultation" / "part2-model-setup.csv"
PART2_RESULTS_DIR = RESULTS_DIR / "part2-interactive-llm-forecasting"
PART2_METRICS_PATH = PART2_RESULTS_DIR / "metrics-part2-interactive-llm-forecasting.csv"
PART2_SUMMARY_DIR = PART2_RESULTS_DIR / "summary-tables"
PART2_SUMMARY_BY_LLM_PATH = PART2_SUMMARY_DIR / "part2-summary-by-llm.csv"
PART2_SUMMARY_BY_LLM_BRANCH_PATH = PART2_SUMMARY_DIR / "part2-summary-by-llm-branch.csv"
PART2_SUMMARY_BY_DATASET_SCENARIO_PATH = PART2_SUMMARY_DIR / "part2-summary-by-dataset-scenario.csv"
PART2_RUN_COVERAGE_PATH = PART2_SUMMARY_DIR / "part2-run-coverage.csv"
PART2_PAIRED_SUMMARY_OVERALL_PATH = PART2_SUMMARY_DIR / "part2-paired-summary-vs-base-overall.csv"
PART2_PAIRED_SUMMARY_BY_LLM_PATH = PART2_SUMMARY_DIR / "part2-paired-summary-vs-base-by-llm.csv"
PART2_VS_PART0_PATH = PART2_SUMMARY_DIR / "part2-vs-part0-comparison.csv"
PART2_VS_PART0_BY_LLM_PATH = PART2_SUMMARY_DIR / "part2-vs-part0-comparison-by-llm.csv"
PART2_VS_PART0_DM_PATH = PART2_SUMMARY_DIR / "part2-vs-part0-diebold-mariano.csv"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "visualizations"
PRE_RESULTS_DIRNAME = "pre-results"
POST_RESULTS_DIRNAME = "post-results"

DATASET_ORDER = ["AirPassengers", "ETTh1", "ILINet", "IceCreamHeater", "Temperature"]
SCENARIO_ORDER = ["S1", "S2", "S3", "S4"]
BRANCH_ORDER = ["Base", "ChalML", "ChalDL"]
METRICS = ["mae", "rmse", "smape", "mase", "r2"]
PRIMARY_PLOT_METRICS = ["mae", "rmse", "r2"]
LOWER_IS_BETTER = {"mae": True, "rmse": True, "smape": True, "mase": True, "r2": False}
METRIC_LABELS = {
    "mae": "MAE",
    "rmse": "RMSE",
    "smape": "sMAPE",
    "mase": "MASE",
    "r2": "R2",
}
DATASET_FILE_MAP = {
    "AirPassengers": "AirPassengers.csv",
    "ETTh1": "ETTh1.csv",
    "ILINet": "ILINet.csv",
    "IceCreamHeater": "IceCreamHeater.csv",
    "Temperature": "Temperature.csv",
}
DATE_COLUMN_MAP = {
    "AirPassengers": "Month",
    "ETTh1": "date",
    "ILINet": "DATE",
    "IceCreamHeater": "Month",
    "Temperature": "Date",
}
BRANCH_COLORS = {
    "Base": "#33658A",
    "ChalML": "#D95D39",
    "ChalDL": "#2A9D8F",
}
SCENARIO_COLORS = {
    "S1": "#355070",
    "S2": "#6D597A",
    "S3": "#B56576",
    "S4": "#E56B6F",
}

PRE_RESULT_OUTPUTS = [
    "pre-results/overall_research_pipeline.png",
    "pre-results/dataset_description_table.csv",
    "pre-results/scenario_protocol_table.csv",
    "pre-results/experimental_system_architecture.png",
]
POST_RESULT_OUTPUTS = [
    "post-results/summary_by_llm.csv",
    "post-results/summary_by_llm_branch.csv",
    "post-results/summary_by_dataset_scenario.csv",
    "post-results/run_coverage.csv",
    "post-results/coverage_failure.png",
    "post-results/paired_summary_vs_base_overall.csv",
    "post-results/paired_summary_vs_base_by_llm.csv",
    "post-results/best_llm_by_dataset_scenario.csv",
    "post-results/part2_vs_part0_comparison.csv",
    "post-results/part2_vs_part0_dm.csv",
    "post-results/part2_vs_part0_delta_rmse.png",
    "post-results/dm_test_scatter.png",
    "post-results/scenario_metric_trend.png",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["pre", "post", "all"],
        default="all",
        help="Which asset group to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for generated visualization assets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned inputs and outputs without creating files.",
    )
    args = parser.parse_args()

    if args.dry_run:
        print_dry_run_report(args.phase, args.output_dir)
        return

    if args.phase in {"pre", "all"}:
        generate_pre_result_assets(args.output_dir / PRE_RESULTS_DIRNAME)
    if args.phase in {"post", "all"}:
        generate_post_result_assets(args.output_dir / POST_RESULTS_DIRNAME)


def print_dry_run_report(phase: str, output_dir: Path) -> None:
    print("Dry-run only. No files will be created.")
    print(f"Output root: {output_dir}")
    print("")

    if phase in {"pre", "all"}:
        print("[pre-results] required inputs")
        for path in [PART1_SETUP_PATH, *[DATA_DIR / filename for filename in DATASET_FILE_MAP.values()]]:
            print(f"  {'OK' if path.exists() else 'MISSING'}  {path.relative_to(BASE_DIR)}")
        print("[pre-results] planned outputs")
        for relative in PRE_RESULT_OUTPUTS:
            print(f"  {relative}")
        print("")

    if phase in {"post", "all"}:
        print("[post-results] required inputs")
        for path in [
            PART2_METRICS_PATH,
            PART2_VS_PART0_PATH,
            PART2_VS_PART0_DM_PATH,
        ]:
            print(f"  {'OK' if path.exists() else 'MISSING'}  {path.relative_to(BASE_DIR)}")
        print("[post-results] optional inputs (rebuilt from metrics when absent)")
        for path in [
            PART2_SUMMARY_BY_LLM_PATH,
            PART2_SUMMARY_BY_LLM_BRANCH_PATH,
            PART2_SUMMARY_BY_DATASET_SCENARIO_PATH,
            PART2_RUN_COVERAGE_PATH,
            PART2_PAIRED_SUMMARY_OVERALL_PATH,
            PART2_PAIRED_SUMMARY_BY_LLM_PATH,
            PART2_VS_PART0_BY_LLM_PATH,
        ]:
            print(f"  {'OK' if path.exists() else 'MISSING'}  {path.relative_to(BASE_DIR)}")
        print("[post-results] planned outputs")
        for relative in POST_RESULT_OUTPUTS:
            print(f"  {relative}")


def generate_pre_result_assets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    setup = read_csv_required(PART1_SETUP_PATH)
    dataset_table = build_dataset_description_table(setup)
    scenario_table = build_scenario_protocol_table()

    draw_overall_research_pipeline(output_dir / "overall_research_pipeline.png")
    save_csv_asset(dataset_table, output_dir / "dataset_description_table.csv")
    save_csv_asset(scenario_table, output_dir / "scenario_protocol_table.csv")
    draw_experimental_system_architecture(output_dir / "experimental_system_architecture.png")


def generate_post_result_assets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_successful_metrics()
    summary_by_llm = load_or_build_summary(
        PART2_SUMMARY_BY_LLM_PATH,
        metrics,
        ["llm_version"],
    )
    summary_by_llm_branch = load_or_build_summary(
        PART2_SUMMARY_BY_LLM_BRANCH_PATH,
        metrics,
        ["llm_version", "branch"],
    )
    summary_by_dataset_scenario = load_or_build_summary(
        PART2_SUMMARY_BY_DATASET_SCENARIO_PATH,
        metrics,
        ["dataset", "scenario"],
    )
    run_coverage = read_csv_optional(PART2_RUN_COVERAGE_PATH)
    paired_summary_overall = read_csv_optional(PART2_PAIRED_SUMMARY_OVERALL_PATH)
    paired_summary_by_llm = read_csv_optional(PART2_PAIRED_SUMMARY_BY_LLM_PATH)
    best_llm_by_dataset_scenario = build_best_llm_by_dataset_scenario(metrics)
    comparison = read_csv_required(PART2_VS_PART0_PATH)
    dm_comparison = read_csv_required(PART2_VS_PART0_DM_PATH)

    save_csv_asset(summary_by_llm, output_dir / "summary_by_llm.csv")
    save_csv_asset(summary_by_llm_branch, output_dir / "summary_by_llm_branch.csv")
    save_csv_asset(summary_by_dataset_scenario, output_dir / "summary_by_dataset_scenario.csv")
    if not run_coverage.empty:
        save_csv_asset(run_coverage, output_dir / "run_coverage.csv")
        draw_coverage_failure_plot(run_coverage, output_dir / "coverage_failure.png")
    if not paired_summary_overall.empty:
        save_csv_asset(paired_summary_overall, output_dir / "paired_summary_vs_base_overall.csv")
    if not paired_summary_by_llm.empty:
        save_csv_asset(paired_summary_by_llm, output_dir / "paired_summary_vs_base_by_llm.csv")
    save_csv_asset(best_llm_by_dataset_scenario, output_dir / "best_llm_by_dataset_scenario.csv")
    save_csv_asset(comparison, output_dir / "part2_vs_part0_comparison.csv")
    save_csv_asset(dm_comparison, output_dir / "part2_vs_part0_dm.csv")

    draw_part2_vs_part0_delta_heatmap(
        comparison,
        "rmse",
        output_dir / "part2_vs_part0_delta_rmse.png",
    )

    draw_dm_test_scatter(dm_comparison, output_dir / "dm_test_scatter.png")
    draw_scenario_metric_trend(metrics, output_dir / "scenario_metric_trend.png")


def build_dataset_description_table(setup: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        setup[["dataset", "dataset_type", "dataset_primary_target"]]
        .drop_duplicates()
        .set_index("dataset")
    )
    rows = []
    for dataset in DATASET_ORDER:
        path = DATA_DIR / DATASET_FILE_MAP[dataset]
        frame = read_csv_required(path)
        total_missing = int(frame.isna().sum().sum())
        total_cells = int(frame.size)
        rows.append(
            {
                "dataset": dataset,
                "series_type": metadata.loc[dataset, "dataset_type"],
                "num_variables": len(frame.columns) - 1,
                "num_samples": len(frame),
                "target_column": metadata.loc[dataset, "dataset_primary_target"],
                "seasonal_period": infer_seasonal_period(setup, dataset),
                "missing_values": total_missing,
                "missing_rate_pct": round(total_missing / total_cells * 100, 2),
                "note": dataset_note(dataset, len(frame), total_missing),
            }
        )
    return pd.DataFrame(rows)


def build_scenario_protocol_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "S1",
                "forecast_mode": "One-step",
                "update_mode": "Static",
                "ground_truth_available": "No",
                "retraining": "No",
                "inference_style": "Recursive inference",
            },
            {
                "scenario": "S2",
                "forecast_mode": "One-step",
                "update_mode": "Rolling",
                "ground_truth_available": "Yes",
                "retraining": "Every step",
                "inference_style": "Continuous retraining",
            },
            {
                "scenario": "S3",
                "forecast_mode": "Multi-step",
                "update_mode": "Rolling",
                "ground_truth_available": "Yes",
                "retraining": "Every step",
                "inference_style": "Continuous retraining",
            },
            {
                "scenario": "S4",
                "forecast_mode": "Multi-step",
                "update_mode": "Block-wise rolling",
                "ground_truth_available": "Yes",
                "retraining": "Every block",
                "inference_style": "Periodic retraining",
            },
        ]
    )


def infer_seasonal_period(setup: pd.DataFrame, dataset: str) -> int | str:
    rows = setup[setup["dataset"] == dataset]
    for raw in rows["baseline_hyperparameters"].dropna():
        try:
            params = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if "seasonal_periods" in params:
            return int(params["seasonal_periods"])
        seasonal_order = params.get("seasonal_order")
        if isinstance(seasonal_order, list) and len(seasonal_order) == 4:
            return int(seasonal_order[-1])
    return "N/A"


def dataset_note(dataset: str, num_samples: int, total_missing: int) -> str:
    notes = []
    if dataset == "ETTh1":
        notes.append(f"largest dataset ({num_samples:,} samples)")
    if dataset == "ILINet" and total_missing:
        notes.append("contains missing values")
    return "; ".join(notes)


def load_successful_metrics() -> pd.DataFrame:
    metrics = read_csv_required(PART2_METRICS_PATH)
    if "execution_status" not in metrics.columns:
        raise ValueError(f"Missing execution_status column in {PART2_METRICS_PATH}")
    ok = metrics[metrics["execution_status"] == "OK"].copy()
    if ok.empty:
        raise ValueError("No successful Part 2 metric rows found.")
    return ok


def load_or_build_summary(
    path: Path,
    metrics: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    grouped = metrics.groupby(group_cols, dropna=False)[METRICS].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    return grouped


def build_rank_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    llm_means = (
        metrics.groupby(["dataset", "scenario", "llm_version"], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )
    rows = []
    for metric in METRICS:
        ascending = LOWER_IS_BETTER[metric]
        ranked = llm_means.copy()
        ranked["rank"] = (
            ranked.groupby(["dataset", "scenario"], dropna=False)[metric]
            .rank(method="min", ascending=ascending)
        )
        counts = (
            ranked[ranked["rank"].isin([1, 2, 3])]
            .groupby(["llm_version", "rank"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        counts["metric"] = metric
        rows.append(counts)
    if not rows:
        return pd.DataFrame(columns=["llm_version", "rank", "count", "metric"])
    return pd.concat(rows, ignore_index=True)


def build_best_llm_by_dataset_scenario(metrics: pd.DataFrame) -> pd.DataFrame:
    means = (
        metrics.groupby(["dataset", "scenario", "llm_version"], dropna=False)[PRIMARY_PLOT_METRICS]
        .mean()
        .reset_index()
    )
    rows = []
    for (dataset, scenario), group in means.groupby(["dataset", "scenario"], dropna=False):
        row = {"dataset": dataset, "scenario": scenario}
        for metric in PRIMARY_PLOT_METRICS:
            idx = group[metric].idxmin() if LOWER_IS_BETTER[metric] else group[metric].idxmax()
            best = group.loc[idx]
            row[f"best_llm_{metric}"] = best["llm_version"]
            row[f"best_{metric}"] = best[metric]
        rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(
        by=["dataset", "scenario"],
        key=lambda series: series.map({name: idx for idx, name in enumerate(DATASET_ORDER)})
        if series.name == "dataset"
        else series.map({name: idx for idx, name in enumerate(SCENARIO_ORDER)}),
    ).reset_index(drop=True)


def build_win_rates(metrics: pd.DataFrame) -> pd.DataFrame:
    llm_means = (
        metrics.groupby(["dataset", "scenario", "llm_version"], dropna=False)[METRICS]
        .mean()
        .reset_index()
    )
    rows = []
    for metric in METRICS:
        ascending = LOWER_IS_BETTER[metric]
        ranked = llm_means.copy()
        ranked["rank"] = (
            ranked.groupby(["dataset", "scenario"], dropna=False)[metric]
            .rank(method="min", ascending=ascending)
        )
        opportunities = (
            ranked.groupby("llm_version", dropna=False)
            .size()
            .reset_index(name="opportunities")
        )
        wins = (
            ranked[ranked["rank"] == 1]
            .groupby("llm_version", dropna=False)
            .size()
            .reset_index(name="wins")
        )
        metric_rates = opportunities.merge(wins, on="llm_version", how="left").fillna({"wins": 0})
        metric_rates["metric"] = metric
        rows.append(metric_rates)
    if not rows:
        return pd.DataFrame(columns=["llm_version", "opportunities", "wins", "metric"])
    return pd.concat(rows, ignore_index=True)


def save_table_assets(df: pd.DataFrame, output_stem: Path, title: str) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    draw_table_image(df, output_stem.with_suffix(".png"), title)


def save_csv_asset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def draw_table_image(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt = get_pyplot()
    display_df = format_table_for_display(df)
    n_rows, n_cols = display_df.shape
    fig_width = max(10, n_cols * 1.55)
    fig_height = max(2.4, (n_rows + 2) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if row == 0:
            cell.set_facecolor("#1F2937")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
    ax.set_title(title, fontsize=13, weight="bold", pad=16)
    fig.tight_layout()
    save_figure(fig, output_path)


def format_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[column] = display[column].astype(str)
    return display


def draw_overall_research_pipeline(output_path: Path) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    draw_box(ax, 0.4, 2.0, 2.1, 1.1, "Part 0\nTraditional Baselines", "#DCEAF7")
    draw_box(ax, 3.0, 2.0, 2.1, 1.1, "Part 1\nLLM Consultation", "#E6F4EA")
    draw_box(ax, 5.6, 2.0, 2.0, 1.1, "Part 2\nInteractive LLM\nForecasting", "#FDEBD0")
    draw_arrow(ax, 2.5, 2.55, 3.0, 2.55)
    draw_arrow(ax, 5.1, 2.55, 5.6, 2.55)

    turn_x = [7.95, 9.0, 10.05, 11.1]
    turn_labels = ["Turn 0\nInspect", "Turn 1\nPreprocess", "Turn 2\nUnderstand", "Turn 3\nForecast"]
    for index, (x, label) in enumerate(zip(turn_x, turn_labels)):
        draw_box(ax, x, 2.1, 0.82, 0.9, label, "#F7D6B4", fontsize=8.5)
        if index == 0:
            draw_arrow(ax, 7.6, 2.55, x, 2.55)
        else:
            draw_arrow(ax, turn_x[index - 1] + 0.82, 2.55, x, 2.55)

    ax.text(
        6.6,
        0.9,
        "Study scope ends at Part 2; Part 2 is executed as a four-turn interaction loop.",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    ax.set_title("Overall Research Pipeline", fontsize=16, weight="bold", pad=16)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_train_test_split(dataset_table: pd.DataFrame, output_path: Path) -> None:
    plt = get_pyplot()
    table = dataset_table.copy()
    table["train_samples"] = (table["num_samples"] * 0.8).astype(int)
    table["test_samples"] = table["num_samples"] - table["train_samples"]
    table = table.set_index("dataset").loc[DATASET_ORDER].reset_index()

    fig, ax = plt.subplots(figsize=(11, 5.8))
    positions = np.arange(len(table))
    ax.barh(positions, table["train_samples"], color="#33658A", label="Train (80%)")
    ax.barh(
        positions,
        table["test_samples"],
        left=table["train_samples"],
        color="#F4A261",
        label="Test (20%)",
    )
    for idx, row in table.iterrows():
        ax.text(row["train_samples"] / 2, idx, f"{row['train_samples']:,}", ha="center", va="center", color="white", fontsize=9)
        ax.text(
            row["train_samples"] + row["test_samples"] / 2,
            idx,
            f"{row['test_samples']:,}",
            ha="center",
            va="center",
            color="#111827",
            fontsize=9,
        )
    ax.set_yticks(positions, table["dataset"])
    ax.invert_yaxis()
    ax.set_xlabel("Number of samples")
    ax.set_title("Chronological 80/20 Train-Test Split", fontsize=15, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_experimental_system_architecture(output_path: Path) -> None:
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    labels = [
        ("Raw CSV", "#E8F0FE"),
        ("Prompt\nInjection", "#E6F4EA"),
        ("LLM\nOutput", "#FFF4E5"),
        ("Parser", "#FCE8E6"),
        ("Executor", "#F3E8FF"),
        ("Metrics", "#E0F2F1"),
        ("Summary\nTables", "#E8EAED"),
    ]
    xs = [0.3, 2.0, 3.8, 5.6, 7.1, 8.9, 10.7]
    widths = [1.35, 1.45, 1.35, 1.15, 1.25, 1.25, 1.55]
    for index, ((label, color), x, width) in enumerate(zip(labels, xs, widths)):
        draw_box(ax, x, 2.0, width, 1.0, label, color)
        if index > 0:
            draw_arrow(ax, xs[index - 1] + widths[index - 1], 2.5, x, 2.5)

    draw_box(ax, 3.95, 0.55, 2.1, 0.82, "Prompt Templates", "#F8FAFC", fontsize=9)
    draw_box(ax, 6.75, 0.55, 2.55, 0.82, "Forecast Scripts / Lists", "#F8FAFC", fontsize=9)
    draw_arrow(ax, 5.0, 1.37, 4.45, 2.0)
    draw_arrow(ax, 8.0, 1.37, 7.7, 2.0)

    ax.set_title("Experimental System Architecture", fontsize=16, weight="bold", pad=16)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_performance_heatmap(metrics: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt = get_pyplot()
    table = (
        metrics.groupby(["dataset", "scenario"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    pivot = table.pivot(index="dataset", columns="scenario", values=metric)
    pivot = pivot.reindex(index=DATASET_ORDER, columns=SCENARIO_ORDER)

    fig, ax = plt.subplots(figsize=(7.8, 5.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="viridis_r" if LOWER_IS_BETTER[metric] else "viridis")
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=15, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title(f"Mean {METRIC_LABELS[metric]} by Dataset and Scenario", fontsize=14, weight="bold")
    annotate_heatmap(ax, pivot.to_numpy(dtype=float))
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(METRIC_LABELS[metric])
    save_figure(fig, output_path)


def draw_coverage_failure_plot(run_coverage: pd.DataFrame, output_path: Path) -> None:
    plt = get_pyplot()
    required = {"dataset", "ok_rows", "failed_rows"}
    if not required.issubset(run_coverage.columns):
        raise ValueError(f"Run coverage table must contain columns: {sorted(required)}")

    frame = run_coverage.copy()
    frame["ok_rows"] = pd.to_numeric(frame["ok_rows"], errors="coerce").fillna(0)
    frame["failed_rows"] = pd.to_numeric(frame["failed_rows"], errors="coerce").fillna(0)
    summary = (
        frame.groupby("dataset", dropna=False)[["ok_rows", "failed_rows"]]
        .sum()
        .reindex(DATASET_ORDER)
        .fillna(0)
    )
    positions = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(positions, summary["ok_rows"], color="#2A9D8F", label="Successful")
    ax.barh(
        positions,
        summary["failed_rows"],
        left=summary["ok_rows"],
        color="#D95D39",
        label="Failed",
    )
    totals = summary["ok_rows"] + summary["failed_rows"]
    for idx, total in enumerate(totals):
        if total > 0:
            fail_rate = summary["failed_rows"].iloc[idx] / total * 100
            ax.text(
                total + max(totals.max() * 0.01, 1),
                idx,
                f"{int(total)} runs, {fail_rate:.1f}% failed",
                va="center",
                fontsize=8.5,
                color="#374151",
            )
    ax.set_yticks(positions, summary.index)
    ax.invert_yaxis()
    ax.set_xlabel("Run count")
    ax.set_title("Part 2 Coverage and Execution Failures", fontsize=14, weight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_llm_comparison_by_scenario(metrics: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt = get_pyplot()
    table = (
        metrics.groupby(["llm_version", "scenario"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    llms = sorted(table["llm_version"].astype(str).unique())
    x = np.arange(len(llms))
    width = 0.18

    fig, ax = plt.subplots(figsize=(max(11, len(llms) * 0.8), 6))
    for idx, scenario in enumerate(SCENARIO_ORDER):
        values = (
            table[table["scenario"] == scenario]
            .set_index("llm_version")
            .reindex(llms)[metric]
            .to_numpy(dtype=float)
        )
        ax.bar(
            x + (idx - 1.5) * width,
            values,
            width=width,
            label=scenario,
            color=SCENARIO_COLORS[scenario],
        )
    ax.set_xticks(x, llms, rotation=35, ha="right")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"LLM Comparison by Scenario ({METRIC_LABELS[metric]})", fontsize=14, weight="bold")
    ax.legend(title="Scenario", ncol=4)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_boxplot_by_llm(metrics: pd.DataFrame, metric: str, output_path: Path) -> None:
    plt = get_pyplot()
    llms = sorted(metrics["llm_version"].astype(str).unique())
    values = [
        pd.to_numeric(metrics.loc[metrics["llm_version"] == llm, metric], errors="coerce").dropna().to_numpy()
        for llm in llms
    ]
    fig, ax = plt.subplots(figsize=(max(11, len(llms) * 0.8), 6))
    box = ax.boxplot(values, labels=llms, patch_artist=True, showfliers=True)
    for patch in box["boxes"]:
        patch.set_facecolor("#DCEAF7")
        patch.set_edgecolor("#33658A")
    for median in box["medians"]:
        median.set_color("#D95D39")
        median.set_linewidth(2)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"Distribution of {METRIC_LABELS[metric]} by LLM", fontsize=14, weight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_rank_count_plot(rank_counts: pd.DataFrame, output_path: Path) -> None:
    plt = get_pyplot()
    aggregate = (
        rank_counts.groupby(["llm_version", "rank"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    llms = sorted(aggregate["llm_version"].astype(str).unique())
    x = np.arange(len(llms))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(11, len(llms) * 0.8), 6))
    colors = {1: "#2A9D8F", 2: "#E9C46A", 3: "#F4A261"}
    for idx, rank in enumerate([1, 2, 3]):
        values = (
            aggregate[aggregate["rank"] == rank]
            .set_index("llm_version")
            .reindex(llms)["count"]
            .fillna(0)
            .to_numpy(dtype=float)
        )
        ax.bar(x + (idx - 1) * width, values, width=width, label=f"Rank {rank}", color=colors[rank])
    ax.set_xticks(x, llms, rotation=35, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Top-3 Rank Counts by LLM", fontsize=14, weight="bold")
    ax.legend()
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_win_rate_plot(win_rates: pd.DataFrame, output_path: Path) -> None:
    plt = get_pyplot()
    aggregate = (
        win_rates.groupby("llm_version", dropna=False)[["wins", "opportunities"]]
        .sum()
    )
    win_rate = (aggregate["wins"] / aggregate["opportunities"] * 100).fillna(0).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(max(10, len(win_rate) * 0.75), 5.8))
    ax.bar(win_rate.index.astype(str), win_rate.values, color="#33658A")
    ax.set_ylabel("Win rate across ranking opportunities (%)")
    ax.set_title("Win Rate by LLM", fontsize=14, weight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_part2_vs_part0_delta_heatmap(
    comparison: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    plt = get_pyplot()
    filtered = comparison[comparison["metric"] == metric].copy()
    if filtered.empty:
        raise ValueError(f"No comparison rows found for metric '{metric}'.")
    filtered["row_label"] = (
        filtered["dataset"].astype(str)
        + " / "
        + filtered["scenario"].astype(str)
        + " / "
        + filtered["part0_model"].astype(str)
    )
    pivot = (
        filtered.groupby(["row_label", "branch"], dropna=False)["delta_part2_minus_part0"]
        .mean()
        .unstack("branch")
        .reindex(columns=BRANCH_ORDER)
    )
    pivot = pivot.loc[sorted(pivot.index)]
    vmax = np.nanmax(np.abs(pivot.to_numpy(dtype=float)))
    vmax = 1 if not np.isfinite(vmax) or vmax == 0 else vmax
    row_label_width = max((len(str(label)) for label in pivot.index), default=0)
    figure_width = min(14.5, max(10.5, 4.8 + len(pivot.columns) * 1.35 + row_label_width * 0.09))
    annotation_fontsize = 7.5 if len(pivot) <= 14 else 6.6

    fig, ax = plt.subplots(
        figsize=(figure_width, max(6.4, len(pivot) * 0.32)),
        constrained_layout=True,
    )
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title(
        f"Part 2 Minus Part 0 Delta ({METRIC_LABELS[metric]})",
        fontsize=14,
        weight="bold",
    )
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", labelsize=10)
    annotate_heatmap(ax, pivot.to_numpy(dtype=float), fontsize=annotation_fontsize)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Delta {METRIC_LABELS[metric]}")
    save_figure(fig, output_path)


def draw_dm_test_scatter(dm_comparison: pd.DataFrame, output_path: Path) -> None:
    plt = get_pyplot()
    frame = dm_comparison.copy()
    frame["p_value"] = pd.to_numeric(frame["p_value"], errors="coerce")
    frame["mean_loss_diff"] = pd.to_numeric(frame["mean_loss_diff"], errors="coerce")
    frame = frame.dropna(subset=["p_value", "mean_loss_diff"])
    frame["neg_log10_p"] = -np.log10(frame["p_value"].clip(lower=np.finfo(float).tiny))
    colors = frame["part2_better_than_part0"].map({True: "#2A9D8F", False: "#D95D39"}).fillna("#6B7280")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(frame["mean_loss_diff"], frame["neg_log10_p"], c=colors, alpha=0.72, edgecolors="white", linewidths=0.4)
    ax.axvline(0, color="#6B7280", linewidth=1)
    ax.axhline(-math.log10(0.05), color="#111827", linestyle="--", linewidth=1, label="p = 0.05")
    ax.set_xlabel("Mean squared-error loss differential (Part 2 - Part 0)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Diebold-Mariano Tests: Part 2 vs Part 0", fontsize=14, weight="bold")
    ax.legend()
    ax.grid(color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_scenario_metric_trend(metrics: pd.DataFrame, output_path: Path) -> None:
    plt = get_pyplot()
    fig, axes = plt.subplots(1, len(PRIMARY_PLOT_METRICS), figsize=(15, 4.8), sharex=True)
    for ax, metric in zip(axes, PRIMARY_PLOT_METRICS):
        table = (
            metrics.groupby(["scenario", "branch"], dropna=False)[metric]
            .mean()
            .reset_index()
        )
        for branch in BRANCH_ORDER:
            branch_values = (
                table[table["branch"] == branch]
                .set_index("scenario")
                .reindex(SCENARIO_ORDER)[metric]
            )
            ax.plot(
                SCENARIO_ORDER,
                branch_values.to_numpy(dtype=float),
                marker="o",
                linewidth=2,
                label=branch,
                color=BRANCH_COLORS[branch],
            )
        ax.set_title(METRIC_LABELS[metric], weight="bold")
        ax.set_xlabel("Scenario")
        ax.grid(color="#E5E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean metric value")
    axes[-1].legend(title="Branch", loc="best")
    fig.suptitle("Metric Trend Across Scenarios", fontsize=15, weight="bold")
    fig.tight_layout()
    save_figure(fig, output_path)


def draw_box(ax, x: float, y: float, width: float, height: float, label: str, color: str, fontsize: float = 10) -> None:
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1,
        edgecolor="#4B5563",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=fontsize, wrap=True)


def draw_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "#4B5563"},
    )


def annotate_heatmap(ax, values: np.ndarray, fontsize: float = 7.5) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=fontsize, color="white")


def get_pyplot():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.dpi": 140,
        }
    )
    return plt


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=180)
    fig.clf()


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
