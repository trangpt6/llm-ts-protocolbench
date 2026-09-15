from __future__ import annotations

import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
PROMPT_PATH = BASE_DIR / "prompts" / "4scenarios.txt"
PROTOCOL_PATH = BASE_DIR / "results" / "part2-interactive-llm-forecasting" / "summary" / "part2-protocol-details.csv"
METRICS_PATH = BASE_DIR / "results" / "part2-interactive-llm-forecasting" / "metrics-part2-interactive-llm-forecasting.csv"
OUT_DIR = BASE_DIR / "results" / "part2-interactive-llm-forecasting" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_REQUIREMENTS = ["retraining", "ground_truth", "blockwise"]
REQUIREMENT_LABELS = {
    "retraining": "Retraining",
    "ground_truth": "Ground truth usage",
    "blockwise": "Blockwise evaluation",
}


def coerce_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "y", "t"}:
            return True
        if text in {"false", "0", "no", "n", "f", ""}:
            return False
    return bool(value)


def parse_scenario_requirements(prompt_path: Path) -> dict[str, dict[str, object]]:
    text = prompt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=Scenario\s+\d+:)", text.strip())
    requirements: dict[str, dict[str, object]] = {}

    for block in blocks:
        match = re.match(r"Scenario\s+(\d+):\s*(.*)", block.strip(), re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        scenario = f"S{match.group(1)}"
        body = match.group(2).strip().lower()
        req: dict[str, object] = {}

        if "model retraining: not performed" in body or "model retraining: not performed during the forecasting process" in body:
            req["expected_retraining"] = False
        elif "model retraining: performed" in body or "retrain after each" in body or "retrained after each" in body:
            req["expected_retraining"] = True
        else:
            req["expected_retraining"] = None

        if "use of ground truth during inference: disabled" in body or "disabled (no updates" in body:
            req["expected_ground_truth"] = False
        elif "use of ground truth during inference: enabled" in body or "enabled" in body:
            req["expected_ground_truth"] = True
        else:
            req["expected_ground_truth"] = None

        req["expected_blockwise"] = True if "block-wise rolling update" in body or "block size:" in body or "blockwise" in body else None
        req["expected_multistep"] = True if "multi-step ahead prediction" in body or "forecast multiple steps ahead" in body else None
        requirements[scenario] = req

    return requirements


def build_scenario_compliance(protocol_df: pd.DataFrame, scenario_requirements: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for _, row in protocol_df.iterrows():
        scenario = str(row.get("scenario", ""))
        req = scenario_requirements.get(scenario, {})
        t2_values = {
            "retraining": coerce_bool(row.get("t2_understands_retraining")),
            "ground_truth": coerce_bool(row.get("t2_understands_groundtruth")),
            "blockwise": coerce_bool(row.get("t2_understands_blockwise")),
        }
        t3_values = {
            "retraining": coerce_bool(row.get("t3_retraining_implemented")),
            "ground_truth": coerce_bool(row.get("t3_uses_ground_truth")),
            "blockwise": coerce_bool(row.get("t3_correct_block_size")),
        }

        semantic_checks: dict[str, bool] = {}
        implementation_checks: dict[str, bool] = {}
        supported_requirements: list[str] = []
        unsupported_requirements: list[str] = []

        for requirement in SUPPORTED_REQUIREMENTS:
            expected_name = f"expected_{requirement}"
            expected_value = req.get(expected_name)
            if expected_value is None:
                unsupported_requirements.append(requirement)
                continue

            supported_requirements.append(requirement)
            t2_value = t2_values[requirement]
            t3_value = t3_values[requirement]
            semantic_checks[requirement] = (t2_value is not None) and (t2_value == expected_value)
            implementation_checks[requirement] = (t3_value is not None) and (t3_value == expected_value)

        if req.get("expected_multistep") is not None:
            unsupported_requirements.append("multistep")

        semantic_compliance = round(100.0 * sum(semantic_checks.values()) / len(semantic_checks), 1) if semantic_checks else None
        implementation_compliance = round(100.0 * sum(implementation_checks.values()) / len(implementation_checks), 1) if implementation_checks else None
        fully_semantic_compliant = bool(semantic_compliance == 100.0 and semantic_checks)
        fully_implementation_compliant = bool(implementation_compliance == 100.0 and implementation_checks)

        row_data = {
            "file_id": row.get("file_id"),
            "dataset": row.get("dataset"),
            "scenario": scenario,
            "branch": row.get("branch"),
            "llm_version": row.get("llm_version"),
            "run_id": row.get("run_id"),
            "expected_requirements": "; ".join(f"{name}={req.get(f'expected_{name}')}" for name in ["retraining", "ground_truth", "blockwise", "multistep"]),
            "supported_requirements": ", ".join(supported_requirements),
            "unsupported_requirements": ", ".join(unsupported_requirements),
            "turn2_semantic_values": json.dumps(t2_values, sort_keys=True),
            "turn3_implementation_values": json.dumps(t3_values, sort_keys=True),
            "semantic_compliance": semantic_compliance,
            "implementation_compliance": implementation_compliance,
            "fully_semantic_compliant": fully_semantic_compliant,
            "fully_implementation_compliant": fully_implementation_compliant,
            "t2_semantic_retraining_correct": semantic_checks.get("retraining"),
            "t2_semantic_ground_truth_correct": semantic_checks.get("ground_truth"),
            "t2_semantic_blockwise_correct": semantic_checks.get("blockwise"),
            "t3_implementation_retraining_correct": implementation_checks.get("retraining"),
            "t3_implementation_ground_truth_correct": implementation_checks.get("ground_truth"),
            "t3_implementation_blockwise_correct": implementation_checks.get("blockwise"),
            "expected_retraining": req.get("expected_retraining"),
            "expected_ground_truth": req.get("expected_ground_truth"),
            "expected_blockwise": req.get("expected_blockwise"),
            "expected_multistep": req.get("expected_multistep"),
        }
        rows.append(row_data)
    return pd.DataFrame(rows)


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(metrics_path)
    if "execution_status" in metrics.columns:
        metrics["execution_success"] = metrics["execution_status"].astype(str).str.upper() == "OK"
    else:
        metrics["execution_success"] = None
    return metrics


def add_execution_success(per_run: pd.DataFrame, metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        per_run["execution_success"] = None
        return per_run
    merged = per_run.merge(metrics_df[["file_id", "execution_success"]], on="file_id", how="left")
    return merged


def build_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for scenario, group in per_run.groupby("scenario", sort=True):
        n_runs = int(len(group))
        semantic_compliant_count = int(group["fully_semantic_compliant"].fillna(False).sum())
        implementation_compliant_count = int(group["fully_implementation_compliant"].fillna(False).sum())
        execution_success_count = int(group["execution_success"].fillna(False).sum())

        row = {
            "scenario": scenario,
            "n_runs": n_runs,
            "avg_semantic_compliance_pct": round(float(group["semantic_compliance"].mean()), 1),
            "avg_implementation_compliance_pct": round(float(group["implementation_compliance"].mean()), 1),
            "fully_semantic_compliant_runs": f"{semantic_compliant_count} / {n_runs}",
            "fully_semantic_compliant_pct": round(100.0 * semantic_compliant_count / n_runs, 1) if n_runs else None,
            "fully_implementation_compliant_runs": f"{implementation_compliant_count} / {n_runs}",
            "fully_implementation_compliant_pct": round(100.0 * implementation_compliant_count / n_runs, 1) if n_runs else None,
            "execution_success": f"{execution_success_count} / {n_runs}",
            "execution_success_pct": round(100.0 * execution_success_count / n_runs, 1) if n_runs else None,
            "implementation_given_semantic": None,
            "implementation_given_semantic_pct": None,
            "execution_given_implementation": None,
            "execution_given_implementation_pct": None,
        }

        fully_semantic = group["fully_semantic_compliant"].fillna(False)
        fully_implementation = group["fully_implementation_compliant"].fillna(False)
        semantic_denominator = int(fully_semantic.sum())
        if semantic_denominator:
            implementation_given_semantic_count = int(((fully_semantic & fully_implementation).sum()))
            row["implementation_given_semantic"] = f"{implementation_given_semantic_count} / {semantic_denominator}"
            row["implementation_given_semantic_pct"] = round(100.0 * implementation_given_semantic_count / semantic_denominator, 1)

        implementation_denominator = int(fully_implementation.sum())
        if implementation_denominator:
            execution_given_implementation_count = int(((fully_implementation & group["execution_success"].fillna(False)).sum()))
            row["execution_given_implementation"] = f"{execution_given_implementation_count} / {implementation_denominator}"
            row["execution_given_implementation_pct"] = round(100.0 * execution_given_implementation_count / implementation_denominator, 1)

        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def build_requirement_level_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in per_run.groupby("scenario", sort=True):
        n_runs = int(len(group))
        for requirement in SUPPORTED_REQUIREMENTS:
            expected_series = group[f"expected_{requirement}"]
            expected_values = expected_series.dropna().unique().tolist()
            supported = bool(expected_values)
            expected_value = expected_values[0] if supported else None
            if supported:
                turn2_col = f"t2_semantic_{requirement}_correct"
                turn3_col = f"t3_implementation_{requirement}_correct"
                turn2_true = int(group[turn2_col].fillna(False).sum())
                turn3_true = int(group[turn3_col].fillna(False).sum())
                scenario_success_true = int(((group[turn2_col].fillna(False)) & (group[turn3_col].fillna(False))).sum())
                turn2_pct = round(100.0 * turn2_true / n_runs, 1) if n_runs else None
                turn3_pct = round(100.0 * turn3_true / n_runs, 1) if n_runs else None
                scenario_success_pct = round(100.0 * scenario_success_true / n_runs, 1) if n_runs else None
            else:
                turn2_true = None
                turn3_true = None
                scenario_success_true = None
                turn2_pct = None
                turn3_pct = None
                scenario_success_pct = None

            rows.append({
                "scenario": scenario,
                "requirement": REQUIREMENT_LABELS[requirement],
                "expected_value": expected_value,
                "supported": supported,
                "turn2_correct_count": turn2_true,
                "turn2_correct_pct": turn2_pct,
                "turn3_correct_count": turn3_true,
                "turn3_correct_pct": turn3_pct,
                "scenario_success_count": scenario_success_true,
                "scenario_success_pct": scenario_success_pct,
            })
    return pd.DataFrame(rows)


def write_outputs(per_run: pd.DataFrame, summary_df: pd.DataFrame, requirement_summary_df: pd.DataFrame, verification_samples: str) -> None:
    per_run.to_csv(OUT_DIR / "scenario_compliance_per_run.csv", index=False)
    summary_df.to_csv(OUT_DIR / "scenario_compliance_summary.csv", index=False)
    requirement_summary_df.to_csv(OUT_DIR / "scenario_requirement_level_summary.csv", index=False)
    (OUT_DIR / "scenario_compliance_summary.md").write_text(build_markdown_summary(summary_df, per_run, requirement_summary_df), encoding="utf-8")
    (OUT_DIR / "scenario_compliance_verification_samples.txt").write_text(verification_samples, encoding="utf-8")


def build_markdown_summary(summary_df: pd.DataFrame, per_run: pd.DataFrame, requirement_summary_df: pd.DataFrame) -> str:
    lines = [
        "# Scenario compliance summary",
        "",
        "This analysis evaluates whether an LLM can understand and execute the forecasting scenario required by the prompt. The unit of analysis is the run, and every metric is computed from explicit numerator/denominator definitions.",
        "",
        "## 1. Metric definitions",
        "",
        "For run $i$, let $R_i$ be the set of supported requirements for that scenario, as inferred from prompts/4scenarios.txt. For each requirement $r \in R_i$, define",
        "",
        "- $S_{i,r} = 1$ if Turn 2 correctly matches the expected value, and $0$ otherwise.",
        "- $I_{i,r} = 1$ if Turn 3 correctly matches the expected value, and $0$ otherwise.",
        "- $SemanticCompliance_i = 100 \times \frac{\sum_{r \in R_i} S_{i,r}}{|R_i|}$.",
        "- $ImplementationCompliance_i = 100 \times \frac{\sum_{r \in R_i} I_{i,r}}{|R_i|}$.",
        "- $FullySemantic_i = 1[SemanticCompliance_i = 100]$.",
        "- $FullyImplementation_i = 1[ImplementationCompliance_i = 100]$.",
        "- $ExecutionSuccess_i = 1$ if the run is marked OK, and $0$ otherwise.",
        "",
        "For a scenario $s$, let $n_s$ be the number of runs in that scenario. The scenario-level metrics are",
        "",
        "- $AverageSemanticCompliance_s = \frac{1}{n_s} \sum_i SemanticCompliance_i$.",
        "- $AverageImplementationCompliance_s = \frac{1}{n_s} \sum_i ImplementationCompliance_i$.",
        "- $FullySemanticRuns_s = \frac{1}{n_s} \sum_i FullySemantic_i$.",
        "- $FullyImplementationRuns_s = \frac{1}{n_s} \sum_i FullyImplementation_i$.",
        "- $ImplementationGivenSemantic_s = \frac{\sum_i 1[FullySemantic_i = 1 \land FullyImplementation_i = 1]}{\sum_i 1[FullySemantic_i = 1]}$.",
        "- $ExecutionGivenImplementation_s = \frac{\sum_i 1[FullyImplementation_i = 1 \land ExecutionSuccess_i = 1]}{\sum_i 1[FullyImplementation_i = 1]}$.",
        "- Requirement-level Turn 2 correctness: $\frac{\sum_i S_{i,r}}{n_s}$.",
        "- Requirement-level Turn 3 correctness: $\frac{\sum_i I_{i,r}}{n_s}$.",
        "- Requirement-level scenario success: $\frac{\sum_i 1[S_{i,r}=1 \land I_{i,r}=1]}{n_s}$.",
        "",
        "## 2. Scenario-to-parser mapping",
        "",
        "| Scenario | Requirement | Expected value | Parser field | Turn | Evaluation rule |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for scenario in ["S1", "S2", "S3", "S4"]:
        req = parse_scenario_requirements(PROMPT_PATH).get(scenario, {})
        for requirement, label in [("retraining", "Retraining"), ("ground_truth", "Ground truth"), ("blockwise", "Blockwise")]:
            expected_value = req.get(f"expected_{requirement}")
            if expected_value is None:
                continue
            parser_field = {
                "retraining": "t2_understands_retraining / t3_retraining_implemented",
                "ground_truth": "t2_understands_groundtruth / t3_uses_ground_truth",
                "blockwise": "t2_understands_blockwise / t3_correct_block_size",
            }[requirement]
            turn = "Turn 2" if requirement in {"retraining", "ground_truth", "blockwise"} else "Turn 2"
            lines.append(f"| {scenario} | {label} | {expected_value} | {parser_field} | {turn} / Turn 3 | semantic/implementation field == expected value |")

    lines.extend([
        "",
        "## 3. Scenario-level results",
        "",
        "| Scenario | Runs | Avg semantic compliance | Avg implementation compliance | Fully semantic compliant | Fully implementation compliant | Execution success | Implementation given semantic | Execution given implementation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['scenario']} | {int(row['n_runs'])} | {row['avg_semantic_compliance_pct']:.1f}% | {row['avg_implementation_compliance_pct']:.1f}% | {row['fully_semantic_compliant_runs']} ({row['fully_semantic_compliant_pct']:.1f}%) | {row['fully_implementation_compliant_runs']} ({row['fully_implementation_compliant_pct']:.1f}%) | {row['execution_success']} ({row['execution_success_pct']:.1f}%) | {row['implementation_given_semantic']} ({row['implementation_given_semantic_pct']:.1f}%) | {row['execution_given_implementation']} ({row['execution_given_implementation_pct']:.1f}%) |"
        )

    lines.extend([
        "",
        "## 4. Requirement-level results",
        "",
        "| Scenario | Requirement | Expected value | Supported | Turn 2 correctness | Turn 3 correctness | Scenario success |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ])

    for _, row in requirement_summary_df.iterrows():
        if row["supported"]:
            lines.append(
                f"| {row['scenario']} | {row['requirement']} | {row['expected_value']} | Yes | {int(row['turn2_correct_count'])} / {int(per_run[per_run['scenario'] == row['scenario']].shape[0])} ({row['turn2_correct_pct']:.1f}%) | {int(row['turn3_correct_count'])} / {int(per_run[per_run['scenario'] == row['scenario']].shape[0])} ({row['turn3_correct_pct']:.1f}%) | {int(row['scenario_success_count'])} / {int(per_run[per_run['scenario'] == row['scenario']].shape[0])} ({row['scenario_success_pct']:.1f}%) |"
            )
        else:
            lines.append(
                f"| {row['scenario']} | {row['requirement']} | — | No | n/a | n/a | n/a |"
            )

    lines.extend([
        "",
        "## 5. Unsupported requirements",
        "",
        "The current parser does not expose a dedicated field for multi-step horizon correctness. This requirement is therefore reported as unsupported rather than inferred from unrelated fields.",
        "",
        "## 6. Verification samples",
        "",
        "The script also writes a verification sample file at results/part2-interactive-llm-forecasting/analysis/scenario_compliance_verification_samples.txt. Each entry reports the scenario, expected requirements, Turn 2 values, Turn 3 values, semantic score, implementation score, execution status, and final classification.",
        "",
    ])
    return "\n".join(lines) + "\n"


def plot_compliance_by_scenario(summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(summary_df))
    ax.bar([i - 0.22 for i in x], summary_df["avg_semantic_compliance_pct"], width=0.2, label="Semantic compliance", color="#2A9D8F")
    ax.bar([i for i in x], summary_df["avg_implementation_compliance_pct"], width=0.2, label="Implementation compliance", color="#E76F51")
    ax.bar([i + 0.22 for i in x], summary_df["execution_success_pct"], width=0.2, label="Execution success", color="#264653")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary_df["scenario"].tolist(), rotation=0)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure A: Compliance and execution by scenario")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_a_compliance_by_scenario.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_requirement_accuracy(summary_df: pd.DataFrame) -> None:
    requirement_names = [REQUIREMENT_LABELS[name] for name in SUPPORTED_REQUIREMENTS]
    turn2_values = []
    turn3_values = []
    for name in SUPPORTED_REQUIREMENTS:
        column = f"{name}_turn2_correct_pct"
        if column in summary_df.columns:
            turn2_values.append(summary_df[column].mean())
        else:
            turn2_values.append(float("nan"))
        column = f"{name}_turn3_correct_pct"
        if column in summary_df.columns:
            turn3_values.append(summary_df[column].mean())
        else:
            turn3_values.append(float("nan"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(requirement_names))
    ax.bar([i - 0.16 for i in x], turn2_values, width=0.32, label="Turn 2 correctness", color="#2A9D8F")
    ax.bar([i + 0.16 for i in x], turn3_values, width=0.32, label="Turn 3 correctness", color="#E76F51")
    ax.set_xticks(list(x))
    ax.set_xticklabels(requirement_names, rotation=0)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure B: Requirement-level Turn 2 vs Turn 3 accuracy")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_b_requirement_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flow_diagram(summary_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.6, 2.0, "Scenario requirement", "Prompt-derived expectation"),
        (3.3, 2.0, "Turn 2 semantic", f"Semantic: {summary_df['avg_semantic_compliance_pct'].mean():.1f}%"),
        (6.1, 2.0, "Turn 3 implementation", f"Implementation: {summary_df['avg_implementation_compliance_pct'].mean():.1f}%"),
        (8.9, 2.0, "Execution", f"Execution: {summary_df['execution_success_pct'].mean():.1f}%"),
    ]

    for x, y, title, subtitle in boxes:
        ax.add_patch(plt.Rectangle((x - 0.9, y - 0.7), 1.8, 1.4, facecolor="#F4F1DE", edgecolor="#264653", linewidth=1.3))
        ax.text(x, y + 0.1, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x, y - 0.25, subtitle, ha="center", va="center", fontsize=8)

    for start, end in [(0.6, 3.3), (3.3, 6.1), (6.1, 8.9)]:
        ax.annotate(
            "",
            xy=(end, 2.0),
            xytext=(start + 0.9, 2.0),
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#2A9D8F"),
        )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figure_c_flow_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_verification_samples(per_run: pd.DataFrame) -> str:
    lines = []
    rng = random.Random(7)
    for scenario, group in per_run.groupby("scenario", sort=True):
        sample = group.sample(n=min(3, len(group)), random_state=rng.randint(0, 1000), replace=False)
        for _, row in sample.iterrows():
            semantic_score = row.get("semantic_compliance")
            implementation_score = row.get("implementation_compliance")
            final_classification = "semantic_ok/implementation_ok" if row.get("fully_semantic_compliant") and row.get("fully_implementation_compliant") else "semantic_ok/implementation_fail"
            if not row.get("fully_semantic_compliant"):
                final_classification = "semantic_fail/implementation_unknown"
            if row.get("execution_success") is True:
                final_classification += "/execution_ok"
            else:
                final_classification += "/execution_fail"
            lines.append(
                f"Scenario={scenario}\n"
                f"run_id={row.get('run_id')}\n"
                f"expected_requirements={row.get('expected_requirements')}\n"
                f"turn2_semantic_values={row.get('turn2_semantic_values')}\n"
                f"turn3_implementation_values={row.get('turn3_implementation_values')}\n"
                f"semantic_score={semantic_score}%\n"
                f"implementation_score={implementation_score}%\n"
                f"execution_status={row.get('execution_success')}\n"
                f"final_classification={final_classification}\n"
            )
    return "\n".join(lines)


def main() -> None:
    protocol = pd.read_csv(PROTOCOL_PATH)
    metrics = load_metrics(METRICS_PATH)
    scenario_requirements = parse_scenario_requirements(PROMPT_PATH)
    per_run = build_scenario_compliance(protocol, scenario_requirements)
    per_run = add_execution_success(per_run, metrics)
    summary = build_summary(per_run)
    requirement_summary = build_requirement_level_summary(per_run)
    verification_samples = build_verification_samples(per_run)
    write_outputs(per_run, summary, requirement_summary, verification_samples)
    plot_compliance_by_scenario(summary)
    plot_requirement_accuracy(summary)
    plot_flow_diagram(summary)
    print(f"Wrote compliance outputs to {OUT_DIR}")
    print(summary.to_string(index=False))
    print(requirement_summary.to_string(index=False))


if __name__ == "__main__":
    main()
