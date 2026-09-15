"""
Generate publication-ready figures and tables for the IJF paper.

Output structure:
    results/paper-figures/
        figures/          # 300 DPI PNG + PDF vector versions
        tables/           # LaTeX-ready CSV tables

Usage:
    python code/part2_7_generate_paper_figures.py
    python code/part2_7_generate_paper_figures.py --format pdf  # vector only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from scipy.stats import pointbiserialr

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
PART2_DIR = RESULTS_DIR / "part2-interactive-llm-forecasting"
SUMMARY_DIR = PART2_DIR / "summary-tables"
JOURNAL_CSV = RESULTS_DIR / "journal-analysis" / "csv"
OUTPUT_DIR = RESULTS_DIR / "paper-figures"
FIG_DIR = OUTPUT_DIR / "figures"
TAB_DIR = OUTPUT_DIR / "tables"

# ── Style constants ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "legend.fontsize": 18,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Three separate palettes so the same colour never carries two different
# meanings across figures (e.g. "success" green vs. "ChalDL" green vs.
# "GPT-5.5" green, which the shared dict previously conflated).
STATUS_COLORS = {"success": "#2A9D8F", "failure": "#E76F51"}
BRANCH_COLORS = {"Base": "#33658A", "ChalML": "#D95D39", "ChalDL": "#6A4C93"}
LLM_COLORS = {
    "ClaudeOpus47": "#33658A", "GPT55": "#588B8B", "Gemini31Pro": "#D95D39",
    "DeepSeekV4Pro": "#E9C46A", "Grok43": "#9B5DE5", "KimiK26": "#F4A261",
}
# Backward-compatible merged dict, kept only for any external references.
COLORS = {**STATUS_COLORS, **BRANCH_COLORS, **LLM_COLORS}
LLM_ORDER = ["ClaudeOpus47", "GPT55", "Gemini31Pro", "DeepSeekV4Pro", "Grok43", "KimiK26"]
# Display names, spelled exactly as in the manuscript (Table V).
LLM_LABEL = {
    "ClaudeOpus47": "Claude Opus 4.7", "GPT55": "GPT-5.5",
    "Gemini31Pro": "Gemini 3.1 Pro", "DeepSeekV4Pro": "DeepSeek-V4 Pro",
    "Grok43": "Grok-4.3", "KimiK26": "Kimi K2.6",
}
BRANCH_ORDER = ["Base", "ChalML", "ChalDL"]
DATASET_ORDER = ["AirPassengers", "ETTh1", "ILINet", "IceCreamHeater", "Temperature"]
SCENARIO_ORDER = ["S1", "S2", "S3", "S4"]


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def load_coverage() -> pd.DataFrame:
    return read_csv(SUMMARY_DIR / "part2-run-coverage.csv")


def load_metrics() -> pd.DataFrame:
    df = read_csv(PART2_DIR / "metrics-part2-interactive-llm-forecasting.csv")
    return df[df["execution_status"] == "OK"].copy()


def load_failure_adjusted_llm() -> pd.DataFrame:
    return read_csv(JOURNAL_CSV / "failure_adjusted_performance_llm_rankings.csv")


def load_failure_adjusted_branch() -> pd.DataFrame:
    return read_csv(JOURNAL_CSV / "failure_adjusted_performance_branch_rankings.csv")


def load_failure_taxonomy_llm() -> pd.DataFrame:
    return read_csv(JOURNAL_CSV / "failure_taxonomy_by_llm.csv")


def load_failure_taxonomy_dataset() -> pd.DataFrame:
    return read_csv(JOURNAL_CSV / "failure_taxonomy_by_dataset.csv")


def load_protocol_execution() -> pd.DataFrame:
    return read_csv(JOURNAL_CSV / "protocol_execution_by_llm.csv")


def load_paired_summary() -> pd.DataFrame:
    return read_csv(SUMMARY_DIR / "part2-paired-summary-vs-base-overall.csv")


def load_wilcoxon() -> pd.DataFrame:
    return read_csv(SUMMARY_DIR / "part2-wilcoxon-vs-base.csv")


def load_win_loss() -> pd.DataFrame:
    return read_csv(SUMMARY_DIR / "part2-vs-part0-win-tie-loss.csv")


def load_scenario_difficulty() -> pd.DataFrame:
    return read_csv(JOURNAL_CSV / "scenario_difficulty_rankings.csv")


def load_dataset_difficulty() -> pd.DataFrame:
    return read_csv(JOURNAL_CSV / "dataset_difficulty_rankings.csv")


# ═══════════════════════════════════════════════════════════════════════
# ETTh1-excluded aggregates (used by Table VI / Fig.4 / Fig.6)
# ═══════════════════════════════════════════════════════════════════════

EXCLUDED_DATASETS = {"ETTh1"}


def standardize_0_1(series: pd.Series, lower_is_better: bool) -> pd.Series:
    """Scale a metric to a 0-1 quality score (mirrors journal_analysis.common)."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    scaled = (s - lo) / (hi - lo)
    return 1.0 - scaled if lower_is_better else scaled


def load_protocol_details() -> pd.DataFrame:
    return read_csv(PART2_DIR / "summary" / "part2-protocol-details.csv")


def protocol_run_level_excl_etth1() -> pd.DataFrame:
    """Run-level protocol rows joined to execution success, ETTh1 excluded."""
    proto = load_protocol_details()
    metrics = read_csv(PART2_DIR / "metrics-part2-interactive-llm-forecasting.csv")[
        ["file_id", "rmse"]
    ].copy()
    metrics["executed_successfully"] = pd.to_numeric(metrics["rmse"], errors="coerce").notna()
    df = proto.merge(metrics[["file_id", "executed_successfully"]], on="file_id", how="left")
    df["executed_successfully"] = df["executed_successfully"].fillna(False)
    return df[~df["dataset"].isin(EXCLUDED_DATASETS)].copy()


def protocol_by_llm_excl_etth1() -> pd.DataFrame:
    """Per-LLM protocol/success aggregates computed on non-ETTh1 runs only."""
    return protocol_run_level_excl_etth1().groupby("llm_version", as_index=False).agg(
        execution_success_rate=("executed_successfully", "mean"),
        overall_protocol_score=("overall_protocol_score", "mean"),
        scenario_understanding_score=("scenario_understanding_score", "mean"),
        code_structure_score=("code_structure_score", "mean"),
    )


def failure_adjusted_llm_excl_etth1() -> pd.DataFrame:
    """Failure-adjusted (execution success × conditional accuracy) on non-ETTh1 runs."""
    cov = load_coverage()
    cov = cov[~cov[cov.columns[0]].isin(EXCLUDED_DATASETS)].copy()
    llm_cov = cov.groupby("llm_version", as_index=False).agg(
        total_rows=("total_rows", "sum"), ok_rows=("ok_rows", "sum")
    )
    llm_cov["execution_success_rate"] = llm_cov["ok_rows"] / llm_cov["total_rows"]

    metrics = load_metrics()
    metrics = metrics[~metrics["dataset"].isin(EXCLUDED_DATASETS)].copy()
    perf = metrics.groupby("llm_version", as_index=False).agg(
        n_success=("rmse", "count"),
        rmse_mean=("rmse", "mean"),
        mase_mean=("mase", "mean"),
        smape_mean=("smape", "mean"),
        r2_mean=("r2", "mean"),
    )
    llm = perf.merge(llm_cov, on="llm_version", how="left")
    llm["rmse_quality"] = standardize_0_1(llm["rmse_mean"], True)
    llm["mase_quality"] = standardize_0_1(llm["mase_mean"], True)
    llm["smape_quality"] = standardize_0_1(llm["smape_mean"], True)
    llm["r2_quality"] = standardize_0_1(llm["r2_mean"], False)
    llm["conditional_accuracy_score"] = llm[
        ["rmse_quality", "mase_quality", "smape_quality", "r2_quality"]
    ].mean(axis=1)
    return llm


# ═══════════════════════════════════════════════════════════════════════
# Save Helpers
# ═══════════════════════════════════════════════════════════════════════

def save_figure(fig, stem: str, formats=("png", "pdf")):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(FIG_DIR / f"{stem}.{fmt}", format=fmt, dpi=300)
    plt.close(fig)


def save_table(df: pd.DataFrame, stem: str):
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB_DIR / f"{stem}.csv", index=False, encoding="utf-8-sig")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Coverage & Failure by Dataset + LLM (two-panel)
# ═══════════════════════════════════════════════════════════════════════

def fig1_coverage_failure():
    cov = load_coverage()

    # ── panel (a): by dataset ──
    ds_data = cov.groupby(
        cov.columns[0]  # dataset column (first col, may have BOM)
    ).agg(total=("total_rows", "sum"), ok=("ok_rows", "sum")).reset_index()
    ds_col = ds_data.columns[0]
    ds_data = ds_data.sort_values("ok", ascending=True)
    ds_data["fail"] = ds_data["total"] - ds_data["ok"]
    ds_data["pct"] = ds_data["ok"] / ds_data["total"] * 100

    # ── panel (b): by LLM ──
    llm_data = cov.groupby("llm_version").agg(
        total=("total_rows", "sum"), ok=("ok_rows", "sum")
    ).reset_index()
    llm_data = llm_data.sort_values("ok", ascending=True)
    llm_data["fail"] = llm_data["total"] - llm_data["ok"]
    llm_data["pct"] = llm_data["ok"] / llm_data["total"] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a)
    y1 = [i * 0.8 for i in range(len(ds_data))]
    ax1.barh(y1, ds_data["ok"], color=STATUS_COLORS["success"], label="Success", height=0.6)
    ax1.barh(y1, ds_data["fail"], left=ds_data["ok"], color=STATUS_COLORS["failure"],
             label="Failed", height=0.6)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(
        [f"{name}\n{ok} ({pct:.1f}%)"
         for name, ok, pct in zip(ds_data[ds_col], ds_data["ok"], ds_data["pct"])],
        fontsize=14,
    )
    ax1.set_xlabel("Number of runs")
    # ax1.set_title("(a) By Dataset", fontweight="bold")

    # Panel (b)
    y2 = [i * 0.8 for i in range(len(llm_data))]
    ax2.barh(y2, llm_data["ok"], color=STATUS_COLORS["success"], label="Success", height=0.6)
    ax2.barh(y2, llm_data["fail"], left=llm_data["ok"], color=STATUS_COLORS["failure"],
             label="Failed", height=0.6)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(
        [f"{LLM_LABEL.get(llm, llm)}\n{ok} ({pct:.1f}%)"
         for llm, ok, pct in zip(llm_data["llm_version"], llm_data["ok"], llm_data["pct"])],
        fontsize=14,
    )
    ax2.set_xlabel("Number of runs")
    # ax2.set_title("(b) By LLM", fontweight="bold")

    # fig.suptitle("Execution Success Rate by Dataset and LLM",
    #              fontweight="bold", fontsize=13, y=1.02)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.58, 0.06), ncol=2, framealpha=0.9, fontsize=13, columnspacing=0.5)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.12, wspace=0.5)
    save_figure(fig, "fig1_execution_success_rate")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Failure-Adjusted Performance (coverage × accuracy tradeoff)
# ═══════════════════════════════════════════════════════════════════════

def fig2_failure_adjusted():
    df = failure_adjusted_llm_excl_etth1()

    fig, ax = plt.subplots(figsize=(8.2, 5.0))

    xs = df["execution_success_rate"].values
    ys = df["conditional_accuracy_score"].values

    # Placement = screen-point offset + anchor corner. Anchoring the text edge
    # (not its centre) at a fixed distance from the marker keeps every label
    # clear of its own point, and of the neighbouring labels, whatever length
    # the LLM name has. Offsets are in points, so they hold if the figure is
    # resized.
    label_placement = {
        "ClaudeOpus47": (-14, 0, "right", "center"),
        "GPT55": (14, 4, "left", "center"),
        "Gemini31Pro": (10, 8, "left", "bottom"),
        "DeepSeekV4Pro": (-18, 0, "right", "center"),
        "Grok43": (15, 9, "left", "center"),
        "KimiK26": (-13, -10, "right", "top"),
    }

    for _, row in df.iterrows():
        llm = row["llm_version"]
        x, y = row["execution_success_rate"], row["conditional_accuracy_score"]
        ax.scatter(x, y, s=300, c=LLM_COLORS.get(llm, "#888888"),
                   edgecolors="white", linewidth=1.5, zorder=5, label=llm)

        dx, dy, ha, va = label_placement.get(llm, (12, 12, "left", "center"))
        ax.annotate(
            LLM_LABEL.get(llm, llm),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=21,
            ha=ha,
            va=va,
        )

    ax.set_xlabel("Execution Success Rate", fontweight="bold")
    ax.set_ylabel("Relative Conditional Accuracy Score", fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.04)
    ax.text(0.95, 0.06, "better →", ha="right", fontsize=22, color="gray",
            transform=ax.transAxes)
    ax.text(0.02, 0.94, "↑ better", fontsize=22, color="gray", transform=ax.transAxes)

    fig.tight_layout()
    save_figure(fig, "fig2_failure_adjusted")

# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Branch Performance vs Base (paired differences)
# ═══════════════════════════════════════════════════════════════════════

def fig3_branch_vs_base():
    """Branch-level forecast accuracy: Base vs ChalML vs ChalDL.

    Two scale-free panels show the distribution of successful runs:
      (a) MASE  - with the MASE = 1 seasonal-naive reference line;
      (b) sMAPE - percentage error.
    Boxplots show the median (line), IQR (box), and 1.5xIQR whiskers.
    """
    metrics = load_metrics()
    for col in ["mase", "smape"]:
        metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

    panels = [
        (axes[0], "mase", "MASE  (lower is better)"),
        (axes[1], "smape", "sMAPE (%)  (lower is better)"),
    ]

    for ax, col, ylabel in panels:
        data = [
            metrics.loc[metrics["branch"] == b, col].dropna().to_numpy(dtype=float)
            for b in BRANCH_ORDER
        ]
        bp = ax.boxplot(
            data,
            labels=BRANCH_ORDER,
            patch_artist=True,
            widths=0.5,
            showfliers=False,
            flierprops=dict(marker="o", markerfacecolor="none", markersize=4,
                            markeredgecolor="#666666", alpha=0.7, linewidth=0.8),
            boxprops=dict(edgecolor="black", linewidth=1.1),
            whiskerprops=dict(color="black", linewidth=1.1),
            capprops=dict(color="black", linewidth=1.1),
            medianprops=dict(color="black", linewidth=2.0),
        )
        for patch, branch in zip(bp["boxes"], BRANCH_ORDER):
            patch.set_facecolor(BRANCH_COLORS[branch])
            patch.set_alpha(0.7)
        if col == "mase":
            ax.axhline(1.0, color="grey", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.text(3.78, 0.90, "MASE = 1", ha="right", va="top",
                    fontsize=15, color="grey")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="both", labelsize=17)
        ax.set_xlim(0.4, 3.82)

        medians = [np.median(d) if len(d) else np.nan for d in data]
        for x, med in enumerate(medians, start=1):
            if np.isfinite(med):
                ax.text(x + 0.28, med, f"{med:.2f}", ha="left", va="center",
                        fontsize=18, fontweight="bold")

    # fig.text(
    #     0.5, 0.03,
    #     "Medians are annotated; whiskers extend to 1.5×IQR; outliers are omitted for readability.",
    #     ha="center", fontsize=19,
    # )
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    save_figure(fig, "fig3_branch_vs_base")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Part 2 vs Part 0 — Win/Loss Summary
# ═══════════════════════════════════════════════════════════════════════

def fig4_part2_vs_part0():
    wl = load_win_loss()
    wl_rmse = wl[wl["metric"] == "rmse"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): Win rate by branch × baseline model (RMSE only)
    ax = axes[0]
    branches = ["Base", "ChalML", "ChalDL"]
    baselines = ["Auto-ARIMA", "Naive (Flat)", "Naive (Seasonal)"]
    x = np.arange(len(baselines))
    width = 0.25

    for i, branch in enumerate(branches):
        rates = []
        for bl in baselines:
            row = wl_rmse[(wl_rmse["branch"] == branch) & (wl_rmse["part0_model"] == bl)]
            if not row.empty:
                rates.append(row["win_rate_pct"].values[0])
            else:
                rates.append(0)
        bars = ax.bar(x + i * width, rates, width, label=branch,
                      color=BRANCH_COLORS.get(branch, "#888888"), edgecolor="white", linewidth=0.8)
        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{rate:.0f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(baselines)
    ax.set_ylabel("RMSE Win Rate (%)")
    # ax.set_title("(a) Part 2 Win Rate vs Traditional Baselines (RMSE)",
    #              fontweight="bold")
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    # Placed above the Auto-ARIMA group (left side), which has no bar value
    # label near y=50-52, instead of over the Naive (Seasonal) group where it
    # previously collided with the ChalML "50%" bar label.
    ax.text(0.02, 0.535, "50% (random chance)", fontsize=7.5, color="gray",
            ha="left", va="bottom", transform=ax.transAxes)
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 105)

    # Panel (b): DM test summary
    ax = axes[1]
    dm_file = SUMMARY_DIR / "part2-vs-part0-dm-win-tie-loss.csv"
    if dm_file.exists():
        dm = read_csv(dm_file)
        dm_agg = dm.groupby("branch").agg(
            sig_wins=("significant_wins", "sum"),
            sig_losses=("significant_losses", "sum"),
        ).reset_index()

        br_list = ["Base", "ChalML", "ChalDL"]
        x2 = np.arange(len(br_list))
        w = 0.35
        wins_vals = [dm_agg[dm_agg["branch"] == b]["sig_wins"].values[0]
                     if b in dm_agg["branch"].values else 0 for b in br_list]
        losses_vals = [dm_agg[dm_agg["branch"] == b]["sig_losses"].values[0]
                       if b in dm_agg["branch"].values else 0 for b in br_list]

        ax.bar(x2 - w/2, wins_vals, w, color=STATUS_COLORS["success"], label="Sig. Wins",
               edgecolor="white", linewidth=0.8)
        ax.bar(x2 + w/2, losses_vals, w, color=STATUS_COLORS["failure"], label="Sig. Losses",
               edgecolor="white", linewidth=0.8)
        for i, (wv, lv) in enumerate(zip(wins_vals, losses_vals)):
            ax.text(i - w/2, wv + max(wins_vals) * 0.02, str(wv),
                    ha="center", fontsize=9, fontweight="bold")
            ax.text(i + w/2, lv + max(losses_vals) * 0.02, str(lv),
                    ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x2)
        ax.set_xticklabels(br_list)
        ax.set_ylabel("Number of DM Tests")
        # ax.set_title("(b) Diebold-Mariano Significant Results (Part 2 vs Part 0)",
        #              fontweight="bold")
        ax.legend(framealpha=0.9)

    # fig.suptitle("Part 2 vs Traditional Baselines: Win Rates & Statistical Tests",
    #              fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig4_part2_vs_part0")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Protocol Compliance vs Execution Success
# ═══════════════════════════════════════════════════════════════════════

def fig5_protocol_vs_success():
    proto = protocol_by_llm_excl_etth1()
    run_level = protocol_run_level_excl_etth1()

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    xs = proto["overall_protocol_score"].astype(float).values
    ys = proto["execution_success_rate"].astype(float).values

    # Placement = screen-point offset + anchor corner, so a label is attached
    # to the side of its own marker that faces free space and cannot grow over
    # it. Claude Opus 4.7 and GPT-5.5 sit close together and take opposite
    # sides; DeepSeek-V4 Pro is the longest name and is the reason the
    # point-biserial box lives in the upper-left instead of the lower-right.
    label_placement = {
        "ClaudeOpus47": (-8, 12, "right", "bottom"),
        "GPT55": (14, 5, "left", "bottom"),
        "Gemini31Pro": (-14, 0, "right", "center"),
        "DeepSeekV4Pro": (12, -9, "left", "top"),
        "Grok43": (5, 8, "right", "bottom"),
        "KimiK26": (18, 2, "left", "bottom"),
    }

    for _, row in proto.iterrows():
        llm = row["llm_version"]
        x, y = row["overall_protocol_score"], row["execution_success_rate"]
        ax.scatter(x, y, s=300, c=LLM_COLORS.get(llm, "#888888"),
                   edgecolors="white", linewidth=1.5, zorder=5)

        dx, dy, ha, va = label_placement.get(llm, (12, 12, "left", "center"))
        ax.annotate(
            LLM_LABEL.get(llm, llm),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=20,
            ha=ha,
            va=va,
        )

    # Đường xu hướng (giữ nguyên)
    from numpy.polynomial.polynomial import polyfit
    if len(xs) > 1:
        b, m = polyfit(xs, ys, 1)
        x_fit = np.linspace(xs.min() - 1.5, xs.max() + 1.5, 50)
        y_fit = b + m * x_fit
        ax.plot(x_fit, y_fit, "k--", alpha=0.55, linewidth=1.5,
                label="LLM-level trend")

    # Box thông tin: point-biserial ở mức run, tính trên dữ liệu loại ETTh1
    x_pb = pd.to_numeric(run_level["overall_protocol_score"], errors="coerce")
    y_pb = run_level["executed_successfully"].astype(int)
    valid = x_pb.notna()
    if valid.sum() > 2 and y_pb[valid].nunique() > 1 and x_pb[valid].nunique() > 1:
        pb_r, pb_p = pointbiserialr(y_pb[valid], x_pb[valid])
        pb_n = int(valid.sum())
        p_str = "p < 0.001" if pb_p < 0.001 else f"p = {pb_p:.3f}"
        # Upper-left, directly under the legend: the lower-right corner has to
        # stay free for the DeepSeek-V4 Pro label.
        ax.text(
            0.02, 0.84,
            f"Point-biserial r = {pb_r:.2f}  ({p_str})\n"
            f"(run-level excl. ETTh1, N = {pb_n})",
            transform=ax.transAxes,
            fontsize=15,
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="lightyellow",
                edgecolor="grey",
                alpha=0.85,
            ),
        )

    ax.set_xlabel("Overall Protocol Score", fontweight="bold")
    ax.set_ylabel("Execution Success Rate", fontweight="bold")
    ax.set_xlim(float(xs.min()) - 3.0, float(xs.max()) + 3.0)
    ax.set_ylim(max(0.0, float(ys.min()) - 0.05), min(1.0, float(ys.max()) + 0.05))

    # Put the one-item trend legend in the open upper-left area so it does
    # not collide with the point-biserial box in the lower-right area.
    ax.legend(loc="upper left", framealpha=0.9, fontsize=15)

    fig.tight_layout()
    save_figure(fig, "fig5_protocol_vs_success")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Failure Taxonomy by LLM (stacked bar)
# ═══════════════════════════════════════════════════════════════════════

def fig6_failure_taxonomy():
    ft = load_failure_taxonomy_llm()

    # Group smaller categories into a catch-all bucket. If the raw taxonomy
    # already contains a genuine category literally named "Other", grouping
    # the remaining tail categories into that same name would silently merge
    # two different things into one bar segment, inflating the real "Other"
    # category. Use a distinct label for the catch-all bucket so the two are
    # never conflated.
    # NOTE: all categories are shown; no grouping into "Other".
    ft["cat_grouped"] = ft["error_category"]

    pivot = ft.pivot_table(
        index="llm_version", columns="cat_grouped", values="failures",
        aggfunc="sum", fill_value=0
    )

    # Keep the same fixed LLM ordering used throughout the paper (LLM_ORDER)
    # for consistency across figures.
    ordered_idx = [llm for llm in LLM_ORDER if llm in pivot.index]
    pivot = pivot.reindex(ordered_idx)

    # Sort columns by total failures (descending) for consistent legend ordering
    pivot = pivot.reindex(pivot.sum().sort_values(ascending=False).index, axis=1)

    colors_list = [
        "#E76F51", "#F4A261", "#E9C46A", "#2A9D8F", "#264653",
        "#6D597A", "#B56576", "#355070", "#999999", "#F94144",
        "#577590", "#43AA8B", "#F9C74F", "#90BE6D", "#F8961E",
        "#277DA1", "#8AB17D", "#B5838D", "#FFB4A2", "#9C89B8",
        "#EF476F", "#118AB2", "#06D6A0", "#FFD166", "#073B4C",
    ]
    cat_colors = dict(zip(pivot.columns, colors_list[:len(pivot.columns)]))

    fig, ax = plt.subplots(figsize=(13.2, 8))
    left = np.zeros(len(pivot))
    y_labels = [LLM_LABEL.get(llm, llm) for llm in pivot.index]
    for cat in pivot.columns:
        vals = pivot[cat].values
        bars = ax.barh(y_labels, vals, left=left, color=cat_colors.get(cat, "#cccccc"),
                       label=cat.replace("_", " ").title(), edgecolor="white", linewidth=0.3)
        for bar, val in zip(bars, vals):
            if val > 3:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(val)), ha="center", va="center", fontsize=15,
                        fontweight="bold")
        left += vals

    ax.set_xlabel("Number of Failures")
    # ax.set_title("Failure Taxonomy by LLM", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right", framealpha=0.9, ncol=2, columnspacing=0.3, fontsize=14)
    fig.tight_layout()
    save_figure(fig, "fig6_failure_taxonomy")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Dataset & Scenario Difficulty Radar / Composite
# ═══════════════════════════════════════════════════════════════════════

def fig7_difficulty_profiles():
    ds = load_dataset_difficulty()
    sc = load_scenario_difficulty()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel (a): Dataset difficulty
    ds_sorted = ds.sort_values("operational_difficulty_score", ascending=True)
    ds_col = ds_sorted.columns[0]  # first column = dataset name
    x1 = range(len(ds_sorted))
    ax1.barh(x1, ds_sorted["operational_difficulty_score"],
             color=["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653"],
             edgecolor="white", linewidth=1, height=0.6)
    for i, (score, cov) in enumerate(zip(ds_sorted["operational_difficulty_score"],
                                          ds_sorted["success_rate"])):
        ax1.text(score + 0.02, i, f"score={score:.3f}\n{cov*100:.0f}% success",
                 va="center", fontsize=8)
    ax1.set_yticks(x1)
    ax1.set_yticklabels(ds_sorted[ds_col])
    ax1.set_xlabel("Operational Difficulty Score\n(lower = easier)")
    # ax1.set_title("(a) Dataset Operational Difficulty", fontweight="bold")
    ax1.set_xlim(0, 1.15)

    # Panel (b): Scenario difficulty
    sc_sorted = sc.sort_values("operational_difficulty_score", ascending=True)
    x2 = range(len(sc_sorted))
    ax2.barh(x2, sc_sorted["operational_difficulty_score"],
             color=["#2A9D8F", "#F4A261", "#E9C46A", "#E76F51"],
             edgecolor="white", linewidth=1, height=0.6)
    for i, (score, cov) in enumerate(zip(sc_sorted["operational_difficulty_score"],
                                          sc_sorted["success_rate"])):
        ax2.text(score + 0.02, i, f"score={score:.3f}\n{cov*100:.0f}% success",
                 va="center", fontsize=8)
    ax2.set_yticks(x2)
    ax2.set_yticklabels(sc_sorted["scenario"])
    ax2.set_xlabel("Operational Difficulty Score\n(lower = easier)")
    # ax2.set_title("(b) Scenario Operational Difficulty", fontweight="bold")
    ax2.set_xlim(0, 0.95)

    # fig.suptitle("Operational Difficulty: Dataset & Scenario Profiles",
    #              fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig7_difficulty_profiles")


# ═══════════════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════════════

def table1_dataset_scenario_design():
    """Combined dataset + scenario design table."""
    import json
    PART1_SETUP = RESULTS_DIR / "part1-llm-strategic-consultation" / "part2-model-setup.csv"
    setup = read_csv(PART1_SETUP)

    # Dataset description
    metadata = (
        setup[["dataset", "dataset_type", "dataset_primary_target"]]
        .drop_duplicates()
        .set_index("dataset")
    )
    rows = []
    for ds_name in DATASET_ORDER:
        path = BASE_DIR / "data" / {
            "AirPassengers": "AirPassengers.csv", "ETTh1": "ETTh1.csv",
            "ILINet": "ILINet.csv", "IceCreamHeater": "IceCreamHeater.csv",
            "Temperature": "Temperature.csv",
        }[ds_name]
        frame = read_csv(path)
        total_missing = int(frame.isna().sum().sum())
        total_cells = int(frame.size)

        # Seasonal period from setup
        ds_setup = setup[setup["dataset"] == ds_name]
        seasonal = "N/A"
        for raw in ds_setup["baseline_hyperparameters"].dropna():
            try:
                params = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if "seasonal_periods" in params:
                seasonal = int(params["seasonal_periods"])
                break
            s_order = params.get("seasonal_order")
            if isinstance(s_order, list) and len(s_order) == 4:
                seasonal = int(s_order[-1])
                break

        train_info = ds_setup[["horizon_or_blocksize", "total_runs"]].iloc[0] if len(ds_setup) > 0 else {}
        rows.append({
            "dataset": ds_name,
            "type": metadata.loc[ds_name, "dataset_type"] if ds_name in metadata.index else "",
            "variables": len(frame.columns) - 1,
            "samples": len(frame),
            "target": metadata.loc[ds_name, "dataset_primary_target"] if ds_name in metadata.index else "",
            "seasonality": seasonal,
            "horizon_or_block": train_info.get("horizon_or_blocksize", ""),
            "total_runs": train_info.get("total_runs", ""),
            "missing_pct": round(total_missing / total_cells * 100, 2),
        })
    ds_table = pd.DataFrame(rows)
    # Round numeric columns
    for col in ["missing_pct"]:
        ds_table[col] = ds_table[col].round(2)
    save_table(ds_table, "table1a_dataset_description")

    # Scenario protocol
    sc_table = pd.DataFrame([
        {"scenario": "S1", "forecast_mode": "One-step", "update_mode": "Static",
         "ground_truth": "No", "retraining": "No", "inference": "Recursive"},
        {"scenario": "S2", "forecast_mode": "One-step", "update_mode": "Rolling",
         "ground_truth": "Yes, each step", "retraining": "Every step", "inference": "Continuous retraining"},
        {"scenario": "S3", "forecast_mode": "Multi-step", "update_mode": "Rolling",
         "ground_truth": "Yes, each step", "retraining": "Every step", "inference": "Continuous retraining"},
        {"scenario": "S4", "forecast_mode": "Multi-step", "update_mode": "Block-wise rolling",
         "ground_truth": "Yes, each block", "retraining": "Every block", "inference": "Periodic retraining"},
    ])
    save_table(sc_table, "table1b_scenario_protocol")


def table2_branch_metrics():
    """Branch-level metrics summary (MASE, RMSE, R2, n)."""
    metrics = load_metrics()
    branch_summary = (
        metrics.groupby(["branch"])
        .agg(
            n=("rmse", "count"),
            mase_mean=("mase", "mean"),
            mase_median=("mase", "median"),
            rmse_mean=("rmse", "mean"),
            r2_mean=("r2", "mean"),
        )
        .reset_index()
    )
    branch_summary = branch_summary.set_index("branch").reindex(BRANCH_ORDER).reset_index()
    # Round for publication
    for col in ["mase_mean", "mase_median", "rmse_mean", "r2_mean"]:
        branch_summary[col] = branch_summary[col].round(3)
    save_table(branch_summary, "table2_branch_metrics")


def table3_paired_comparison():
    """Paired comparison vs Base with Wilcoxon (rounded)."""
    paired = load_paired_summary()
    wilcoxon = load_wilcoxon()

    rows = []
    for _, p_row in paired.iterrows():
        branch = p_row["challenger_branch"]
        for metric in ["mae", "rmse", "mase", "r2"]:
            dif_col = f"{metric}_difference"
            w_row = wilcoxon[(wilcoxon["branch"] == branch) & (wilcoxon["metric"] == metric)]
            p_val = w_row["p_value"].values[0] if not w_row.empty else None
            # Format p-value nicely (values >= 0.001 all use the same 4-decimal
            # format, so a single branch replaces the previous three identical
            # elif/else branches).
            if p_val is None:
                p_str = ""
            elif p_val < 0.001:
                p_str = "< 0.001"
            else:
                p_str = f"{p_val:.4f}"

            rows.append({
                "branch": branch,
                "metric": metric.upper(),
                "n_pairs": int(p_row["n_pairs"]),
                "base_median": round(float(p_row.get(f"{metric}_base_median", 0)), 3),
                "challenger_median": round(float(p_row.get(f"{metric}_challenger_median", 0)), 3),
                "median_delta": round(float(p_row.get(f"{dif_col}_median", 0)), 3),
                "mean_delta": round(float(p_row.get(f"{dif_col}_mean", 0)), 3),
                "wilcoxon_p": p_str,
            })

    df = pd.DataFrame(rows)
    save_table(df, "table3_paired_vs_base")


def table4_win_loss():
    """RMSE win/loss table vs traditional baselines."""
    wl = load_win_loss()
    rmse = wl[wl["metric"] == "rmse"].copy()
    rmse = rmse[["branch", "part0_model", "comparisons", "wins", "losses",
                  "ties", "win_rate_pct"]]
    # Use apply+round to avoid floating-point representation issues
    rmse["win_rate_pct"] = rmse["win_rate_pct"].apply(lambda x: round(float(x), 1))
    save_table(rmse, "table4_win_loss_vs_part0")


def table5_execution_success():
    """Execution success rate by dataset and LLM (rounded)."""
    cov = load_coverage()
    ds_col = cov.columns[0]

    ds_agg = cov.groupby(ds_col).agg(
        total=("total_rows", "sum"), ok=("ok_rows", "sum")
    ).reset_index()
    ds_agg["execution_success_rate_pct"] = (ds_agg["ok"] / ds_agg["total"] * 100).round(2)

    llm_agg = cov.groupby("llm_version").agg(
        total=("total_rows", "sum"), ok=("ok_rows", "sum")
    ).reset_index()
    llm_agg["execution_success_rate_pct"] = (llm_agg["ok"] / llm_agg["total"] * 100).round(2)

    save_table(ds_agg, "table5a_execution_success_by_dataset")
    save_table(llm_agg, "table5b_execution_success_by_llm")


def table6_failure_categories():
    """Failure categories by LLM."""
    ft = load_failure_taxonomy_llm()
    save_table(ft, "table6_failure_taxonomy_by_llm")


def table7_protocol_scores():
    """Protocol scores by LLM (rounded), computed on non-ETTh1 runs only."""
    proto = protocol_by_llm_excl_etth1()
    for col in ["execution_success_rate", "overall_protocol_score",
                "scenario_understanding_score", "code_structure_score"]:
        if col in proto.columns:
            proto[col] = proto[col].apply(lambda x: round(float(x), 2))
    save_table(proto, "table7_protocol_by_llm")


def table8_recommendation_consensus():
    """Part 1 recommendation consensus: model vote & HP vote by dataset×scenario.

    Built from part2-model-setup.csv which records, for each dataset×scenario,
    the top-voted baseline model, its model-family vote share (model_pct),
    and the top hyperparameter configuration vote share (hp_pct).
    """
    setup_path = RESULTS_DIR / "part1-llm-strategic-consultation" / "part2-model-setup.csv"
    setup = read_csv(setup_path)

    tbl = setup[[
        "dataset", "scenario",
        "baseline_model", "model_votes", "model_pct",
        "hp_votes", "hp_pct", "hp_selection_method",
    ]].copy()

    tbl["model_pct"] = tbl["model_pct"].apply(lambda x: round(float(x), 1))
    tbl["hp_pct"] = tbl["hp_pct"].apply(lambda x: round(float(x), 1))

    tbl = tbl.sort_values(
        by=["dataset", "scenario"],
        key=lambda s: s.map({n: i for i, n in enumerate(DATASET_ORDER)})
        if s.name == "dataset" else s.map({n: i for i, n in enumerate(SCENARIO_ORDER)}),
    ).reset_index(drop=True)

    save_table(tbl, "table8_recommendation_consensus")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

FIGURE_GENERATORS = {
    "fig1": fig1_coverage_failure,  # kept internal name; output file is fig1_execution_success_rate
    "fig2": fig2_failure_adjusted,
    "fig3": fig3_branch_vs_base,
    "fig4": fig4_part2_vs_part0,
    "fig5": fig5_protocol_vs_success,
    "fig6": fig6_failure_taxonomy,
    "fig7": fig7_difficulty_profiles,
}

TABLE_GENERATORS = {
    "t1": table1_dataset_scenario_design,
    "t2": table2_branch_metrics,
    "t3": table3_paired_comparison,
    "t4": table4_win_loss,
    "t5": table5_execution_success,
    "t6": table6_failure_categories,
    "t7": table7_protocol_scores,
    "t8": table8_recommendation_consensus,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="*",
                        choices=list(FIGURE_GENERATORS) + ["all"],
                        default=["all"],
                        help="Which figures to generate (default: all)")
    parser.add_argument("--tables", nargs="*",
                        choices=list(TABLE_GENERATORS) + ["all"],
                        default=["all"],
                        help="Which tables to generate (default: all)")
    parser.add_argument("--format", nargs="+", choices=["png", "pdf"],
                        default=["png", "pdf"],
                        help="Output format (default: png pdf)")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # Figures
    if "all" in args.figures:
        figs_to_run = list(FIGURE_GENERATORS)
    else:
        figs_to_run = args.figures

    for key in figs_to_run:
        print(f"[FIGURE] Generating {key}...")
        try:
            FIGURE_GENERATORS[key]()
            print(f"  ✓ {key} done")
        except Exception as e:
            print(f"  ✗ {key} FAILED: {e}")

    # Tables
    if "all" in args.tables:
        tabs_to_run = list(TABLE_GENERATORS)
    else:
        tabs_to_run = args.tables

    for key in tabs_to_run:
        print(f"[TABLE]  Generating {key}...")
        try:
            TABLE_GENERATORS[key]()
            print(f"  ✓ {key} done")
        except Exception as e:
            print(f"  ✗ {key} FAILED: {e}")

    print(f"\nDone! Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
