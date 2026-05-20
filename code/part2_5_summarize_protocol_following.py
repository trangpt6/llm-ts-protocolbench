"""Summarize Part 2 protocol following from the parsed master log.

This script is independent from metric execution.  Run it after
code/part2_2_parse_master_logs.py has produced the master log.  If metrics are
also available, run code/part2_4_execution_failure_summary.py separately for
execution/failure summaries.

Inputs:
    logs/master-logs/master-log-part2-interactive-llm-forecasting.csv
    results/part2-interactive-llm-forecasting/scripts-output/<file_id>.py

Outputs:
    results/part2-interactive-llm-forecasting/summary/part2-protocol-details.csv
    results/part2-interactive-llm-forecasting/summary/part2-protocol-by-llm.csv
    results/part2-interactive-llm-forecasting/summary/part2-protocol-by-dataset.csv
    results/part2-interactive-llm-forecasting/summary/part2-protocol-by-llm-dataset.csv
    results/part2-interactive-llm-forecasting/summary/part2-protocol-by-scenario.csv
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MASTER_LOG_PATH = BASE_DIR / "logs" / "master-logs" / "master-log-part2-interactive-llm-forecasting.csv"
PART2_RESULTS_DIR = BASE_DIR / "results" / "part2-interactive-llm-forecasting"
SUMMARY_DIR = PART2_RESULTS_DIR / "summary"

BRANCH_TO_TRACK = {
    "Base": "baseline",
    "ChalML": "challenger_ml",
    "ChalDL": "challenger_dl",
}

MODEL_KEYWORDS = {
    "sarima": [r"\bsarimax\b", r"\bsarima\b", r"\barima\b"],
    "sarimax": [r"\bsarimax\b"],
    "arima": [r"\barima\b", r"\bsarimax\b"],
    "exponential smoothing": [r"\bexponentialsmoothing\b", r"\bholtwinters\b", r"\bholt-winters\b"],
    "holt-winters": [r"\bexponentialsmoothing\b", r"\bholtwinters\b", r"\bholt-winters\b"],
    "lightgbm": [r"\blgbmregressor\b", r"\blightgbm\b", r"\blgb\b"],
    "xgboost": [r"\bxgbregressor\b", r"\bxgboost\b", r"\bxgb\b"],
    "randomforest": [r"\brandomforestregressor\b", r"\brandom forest\b"],
    "random forest": [r"\brandomforestregressor\b", r"\brandom forest\b"],
    "lstm": [r"\blstm\b", r"\bnn\.lstm\b"],
    "gru": [r"\bgru\b", r"\bnn\.gru\b"],
    "tcn": [r"\btcn\b", r"\bconv1d\b", r"\bdilated"],
    "prophet": [r"\bprophet\b"],
}


def main() -> None:
    if not MASTER_LOG_PATH.exists():
        raise FileNotFoundError(f"Master log not found: {MASTER_LOG_PATH}")

    df = pd.read_csv(MASTER_LOG_PATH)
    details = build_protocol_details(df)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    details.to_csv(SUMMARY_DIR / "part2-protocol-details.csv", index=False, encoding="utf-8-sig")
    write_group_summary(details, ["llm_version"], "part2-protocol-by-llm.csv")
    write_group_summary(details, ["dataset"], "part2-protocol-by-dataset.csv")
    write_group_summary(details, ["llm_version", "dataset"], "part2-protocol-by-llm-dataset.csv")
    write_group_summary(details, ["scenario"], "part2-protocol-by-scenario.csv")
    print(f"Saved Part 2 protocol summaries -> {SUMMARY_DIR}")


def build_protocol_details(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        file_id = Path(str(row.get("chat_log_file", ""))).stem
        output_type = text(row.get("t3_output_type")).upper()
        script = load_script_text(row)
        expected = expected_setup(row)
        checks = protocol_checks(row, script, expected)
        scores = protocol_scores(checks, output_type)

        rows.append({
            "file_id": file_id,
            "dataset": row.get("dataset", ""),
            "scenario": row.get("scenario", ""),
            "branch": row.get("branch", ""),
            "track": BRANCH_TO_TRACK.get(text(row.get("branch")), ""),
            "llm_version": row.get("llm_version", ""),
            "run_id": row.get("run_id", ""),
            "output_type": output_type,
            "expected_model": expected.get("model", ""),
            "expected_hyperparameters": expected.get("hyperparameters", ""),
            **checks,
            **scores,
        })
    return pd.DataFrame(rows)


def protocol_checks(row: pd.Series, script: str, expected: dict[str, str]) -> dict[str, Any]:
    output_type = text(row.get("t3_output_type")).upper()
    t2_model = text(row.get("t2_model_confirmed"))
    t2_hyperparams = text(row.get("t2_hyperparams_confirmed"))
    expected_model = expected.get("model", "")
    expected_hyperparams = expected.get("hyperparameters", "")

    model_source = expected_model or t2_model
    hyperparam_source = expected_hyperparams or t2_hyperparams

    return {
        "t0_format_followed": as_bool(row.get("t0_format_followed")),
        "t1_format_followed": as_bool(row.get("t1_format_followed")),
        "t1_has_previous_summary": as_bool(row.get("t1_has_previous_summary")),
        "t1_overcleaned_flag": as_bool(row.get("t1_overcleaned_flag")),
        "t2_format_followed": as_bool(row.get("t2_format_followed")),
        "t2_has_previous_summary": as_bool(row.get("t2_has_previous_summary")),
        "t2_model_confirmed_nonempty": bool(t2_model),
        "t2_hyperparams_confirmed_nonempty": bool(t2_hyperparams),
        "t2_understands_blockwise": as_bool(row.get("t2_understands_blockwise")),
        "t2_understands_retraining": as_bool(row.get("t2_understands_retraining")),
        "t2_understands_groundtruth": as_bool(row.get("t2_understands_groundtruth")),
        "t3_parse_success": as_bool(row.get("t3_parse_success")),
        "t3_extra_text_flag": as_bool(row.get("t3_extra_text_flag")),
        "t3_has_seed": as_bool(row.get("t3_has_seed")),
        "t3_reads_csv": as_bool(row.get("t3_reads_csv")),
        "t3_correct_split_80_20": as_bool(row.get("t3_correct_split_80_20")),
        "t3_correct_block_size": as_bool(row.get("t3_correct_block_size")),
        "t3_retraining_implemented": as_bool(row.get("t3_retraining_implemented")),
        "t3_uses_ground_truth": as_bool(row.get("t3_uses_ground_truth")),
        "t3_remainder_handled": as_bool(row.get("t3_remainder_handled")),
        "t3_prints_forecasts": as_bool(row.get("t3_prints_forecasts")),
        "code_model_matches_expected": code_model_matches(script, model_source) if output_type == "SCRIPT" else None,
        "code_hyperparams_match_expected": code_hyperparams_match(script, hyperparam_source) if output_type == "SCRIPT" else None,
        "script_uses_expected_family": expected_model_family(script, model_source) if output_type == "SCRIPT" else None,
    }


def protocol_scores(checks: dict[str, Any], output_type: str) -> dict[str, Any]:
    format_items = [
        checks["t0_format_followed"],
        checks["t1_format_followed"],
        checks["t1_has_previous_summary"],
        checks["t2_format_followed"],
        checks["t2_has_previous_summary"],
        not checks["t3_extra_text_flag"],
    ]
    scenario_items = [
        checks["t2_model_confirmed_nonempty"],
        checks["t2_hyperparams_confirmed_nonempty"],
        checks["t2_understands_blockwise"],
        checks["t2_understands_retraining"],
        checks["t2_understands_groundtruth"],
    ]

    if output_type == "SCRIPT":
        code_items = [
            checks["t3_parse_success"],
            checks["t3_has_seed"],
            checks["t3_reads_csv"],
            checks["t3_correct_split_80_20"],
            checks["t3_correct_block_size"],
            checks["t3_retraining_implemented"],
            checks["t3_uses_ground_truth"],
            checks["t3_remainder_handled"],
            checks["t3_prints_forecasts"],
        ]
        model_items = [
            checks["script_uses_expected_family"],
            checks["code_model_matches_expected"],
            checks["code_hyperparams_match_expected"],
        ]
    elif output_type == "LIST":
        code_items = [checks["t3_parse_success"]]
        model_items = []
    else:
        code_items = [False]
        model_items = []

    format_score = score(format_items)
    scenario_score = score(scenario_items)
    code_score = score(code_items)
    model_score = score(model_items)
    turn0_score = score([checks["t0_format_followed"]])
    turn1_score = score([
        checks["t1_format_followed"],
        checks["t1_has_previous_summary"],
        not checks["t1_overcleaned_flag"],
    ])
    turn2_score = score([
        checks["t2_format_followed"],
        checks["t2_has_previous_summary"],
        *scenario_items,
    ])
    turn3_score = score([
        not checks["t3_extra_text_flag"],
        *code_items,
        *model_items,
    ])
    overall_items = [
        format_score,
        scenario_score,
        code_score,
        model_score if model_items else None,
    ]

    return {
        "format_follow_score": format_score,
        "scenario_understanding_score": scenario_score,
        "code_structure_score": code_score,
        "model_hyperparam_score": model_score,
        "turn0_score": turn0_score,
        "turn1_score": turn1_score,
        "turn2_score": turn2_score,
        "turn3_score": turn3_score,
        "overall_protocol_score": average_scores(overall_items),
        "strict_protocol_pass": all(
            value == 100.0
            for value in [turn0_score, turn1_score, turn2_score, turn3_score]
        ),
    }


def expected_setup(row: pd.Series) -> dict[str, str]:
    """
    Best-effort expected setup extraction.

    The script intentionally avoids depending on one exact setup schema.  It
    first tries the Part 2 setup loader used by compute_metrics, then falls back
    to Turn-2 text parsed from the master log.
    """
    dataset = text(row.get("dataset"))
    scenario = text(row.get("scenario"))
    track = BRANCH_TO_TRACK.get(text(row.get("branch")), "baseline")
    fallback = {
        "model": text(row.get("t2_model_confirmed")),
        "hyperparameters": text(row.get("t2_hyperparams_confirmed")),
    }

    try:
        from part2_api import model_setup, paths

        setup = model_setup.load_model_setup(
            paths.DEFAULT_MODEL_SETUP_PATH,
            dataset,
            scenario,
            track,
        )
    except Exception:
        return fallback

    model = first_nonempty_attr(
        setup,
        [
            "model",
            "model_name",
            "fixed_model",
            "algorithm",
            "method",
            "model_type",
        ],
    )
    hyperparams = first_nonempty_attr(
        setup,
        [
            "hyperparameters",
            "hyperparams",
            "fixed_hyperparameters",
            "params",
            "model_params",
            "config",
        ],
    )
    return {
        "model": stringify_setup_value(model) or fallback["model"],
        "hyperparameters": stringify_setup_value(hyperparams) or fallback["hyperparameters"],
    }


def first_nonempty_attr(obj: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value not in (None, "", {}):
                return value
    return ""


def stringify_setup_value(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def code_model_matches(script: str, expected_model: str) -> bool | None:
    if not script or not expected_model:
        return None
    return expected_model_family(script, expected_model)


def expected_model_family(script: str, expected_model: str) -> bool | None:
    expected = normalize_text(expected_model)
    code = normalize_text(script)
    if not expected or not code:
        return None

    for name, patterns in MODEL_KEYWORDS.items():
        if name in expected:
            return any(re.search(pattern, code) for pattern in patterns)
    return None


def code_hyperparams_match(script: str, expected_hyperparams: str) -> bool | None:
    if not script or not expected_hyperparams:
        return None

    expected_tokens = extract_hyperparam_tokens(expected_hyperparams)
    if not expected_tokens:
        return None

    code = normalize_text(script)
    matched = 0
    for token in expected_tokens:
        if token in code:
            matched += 1
    return matched / len(expected_tokens) >= 0.6


def extract_hyperparam_tokens(text_value: str) -> list[str]:
    normalized = normalize_text(text_value)
    tokens: set[str] = set()

    for match in re.finditer(r"\b[a-z_][a-z0-9_]*\s*=\s*[-+]?\d+(?:\.\d+)?", normalized):
        tokens.add(match.group(0).replace(" ", ""))
    for match in re.finditer(r"\b(?:order|seasonal_order)\s*[:=]\s*\(?\s*\d+\s*,\s*\d+\s*,\s*\d+", normalized):
        tokens.add(re.sub(r"\s+", "", match.group(0)))
    for match in re.finditer(r"\b[-+]?\d+(?:\.\d+)?\b", normalized):
        tokens.add(match.group(0))

    noisy = {"0", "1", "2", "3", "4", "5", "10", "12", "24", "30", "52", "80", "100"}
    compact = [token for token in tokens if token not in noisy or len(tokens) <= 5]
    return sorted(compact)


def load_script_text(row: pd.Series) -> str:
    inline = text(row.get("t3_script_content"))
    if inline:
        return inline

    script_file = text(row.get("script_file"))
    if not script_file:
        return ""
    path = resolve_repo_path(script_file)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def resolve_repo_path(value: str) -> Path:
    raw = value.replace("\\", "/")
    path = Path(raw)
    if path.is_absolute():
        return path
    if raw.startswith("llm-ts-protocolbench/"):
        raw = raw.split("/", 1)[1]
    return BASE_DIR / raw


def write_group_summary(df: pd.DataFrame, group_cols: list[str], filename: str) -> None:
    grouped = df.groupby(group_cols, dropna=False)
    summary = grouped.apply(summary_row, include_groups=False).reset_index()
    summary = summary.sort_values(["overall_protocol_score", "total_runs"], ascending=[False, False])
    summary.to_csv(SUMMARY_DIR / filename, index=False, encoding="utf-8-sig")


def summary_row(group: pd.DataFrame) -> pd.Series:
    total = len(group)
    script_rows = int((group["output_type"] == "SCRIPT").sum())
    list_rows = int((group["output_type"] == "LIST").sum())
    error_rows = int((group["output_type"] == "ERROR").sum())
    text_rows = int((group["output_type"] == "TEXT").sum())
    return pd.Series({
        "total_runs": total,
        "script_rows": script_rows,
        "script_rate_pct": pct(script_rows, total),
        "list_rows": list_rows,
        "list_rate_pct": pct(list_rows, total),
        "model_error_rows": error_rows,
        "text_rows": text_rows,
        "format_follow_score": mean_pct(group["format_follow_score"]),
        "scenario_understanding_score": mean_pct(group["scenario_understanding_score"]),
        "code_structure_score": mean_pct(group["code_structure_score"]),
        "model_hyperparam_score": mean_pct(group["model_hyperparam_score"]),
        "turn0_score": mean_pct(group["turn0_score"]),
        "turn1_score": mean_pct(group["turn1_score"]),
        "turn2_score": mean_pct(group["turn2_score"]),
        "turn3_score": mean_pct(group["turn3_score"]),
        "overall_protocol_score": mean_pct(group["overall_protocol_score"]),
        "strict_protocol_pass_rate_pct": bool_rate(group["strict_protocol_pass"]),
        "t0_format_follow_rate_pct": bool_rate(group["t0_format_followed"]),
        "t1_format_follow_rate_pct": bool_rate(group["t1_format_followed"]),
        "t2_format_follow_rate_pct": bool_rate(group["t2_format_followed"]),
        "understands_blockwise_rate_pct": bool_rate(group["t2_understands_blockwise"]),
        "understands_retraining_rate_pct": bool_rate(group["t2_understands_retraining"]),
        "understands_groundtruth_rate_pct": bool_rate(group["t2_understands_groundtruth"]),
        "code_model_match_rate_pct": nullable_bool_rate(group["code_model_matches_expected"]),
        "code_hyperparam_match_rate_pct": nullable_bool_rate(group["code_hyperparams_match_expected"]),
    })


def score(items: list[Any]) -> float | None:
    values = [as_bool(item) for item in items if item is not None]
    if not values:
        return None
    return round(sum(values) / len(values) * 100, 2)


def average_scores(items: list[float | None]) -> float:
    values = [item for item in items if item is not None]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def mean_pct(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    return round(float(values.mean()), 2)


def bool_rate(series: pd.Series) -> float:
    return nullable_bool_rate(series.fillna(False))


def nullable_bool_rate(series: pd.Series) -> float:
    values = [as_bool(value) for value in series if value is not None and str(value) != "nan"]
    if not values:
        return 0.0
    return round(sum(values) / len(values) * 100, 2)


def pct(numerator: Any, denominator: Any) -> float:
    denominator = int(denominator)
    if denominator == 0:
        return 0.0
    return round(float(numerator) / denominator * 100, 2)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("_", " ")).strip()


if __name__ == "__main__":
    main()
