"""Run protocol-aligned traditional baselines for Part 0.

Outputs:
    results/part0-traditional-baselines/baseline-traditional-results.csv
    results/part0-traditional-baselines/forecast-outputs/<dataset>__<scenario>__<model>.csv
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pmdarima as pm

from part2_api.datasets import DatasetBundle, load_dataset
from part2_api.model_setup import load_model_setup
from part2_api.paths import DEFAULT_MODEL_SETUP_PATH
from part2_3_compute_metrics import calculate_metrics, calculate_naive_factor


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "part0-traditional-baselines"
SCENARIOS = ("S1", "S2", "S3", "S4")
DATASETS = ("AirPassengers", "ETTh1", "IceCreamHeater", "ILINet", "Temperature")
MODEL_NAMES = ("Naive (Flat)", "Naive (Seasonal)", "Auto-ARIMA")
ARIMA_TRAIN_LIMIT = 5000
RESULT_COLUMNS = [
    "dataset",
    "model",
    "scenario",
    "horizon_or_block",
    "forecast_length",
    "forecast_output_file",
    "mae",
    "rmse",
    "smape",
    "mase",
    "r2",
]
RESULT_KEY_COLUMNS = ["dataset", "scenario", "model"]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    target_col: str
    seasonal_period: int


DATASET_CONFIGS = {
    "AirPassengers": DatasetConfig("AirPassengers", "Passengers", 12),
    "ETTh1": DatasetConfig("ETTh1", "OT", 24),
    "IceCreamHeater": DatasetConfig("IceCreamHeater", "Ice cream", 12),
    "ILINet": DatasetConfig("ILINet", "% WEIGHTED ILI", 52),
    "Temperature": DatasetConfig("Temperature", "Daily minimum temperatures", 365),
}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    selected_datasets = parse_selection(args.datasets, DATASETS, "dataset")
    selected_scenarios = parse_selection(args.scenarios, SCENARIOS, "scenario")
    selected_models = parse_selection(args.models, MODEL_NAMES, "model")

    rows: list[dict[str, object]] = []
    for dataset_name in selected_datasets:
        config = DATASET_CONFIGS[dataset_name]
        bundle = load_dataset(dataset_name, config.target_col, source="auto")
        print(f"Processing: {dataset_name}")
        for scenario in selected_scenarios:
            setup = load_model_setup(DEFAULT_MODEL_SETUP_PATH, dataset_name, scenario, "baseline")
            horizon_or_block = int(setup.horizon_or_block)
            print(f"  Scenario {scenario} (horizon_or_block={horizon_or_block})")
            for model_name in selected_models:
                forecasts = forecast_for_scenario(
                    bundle=bundle,
                    model_name=model_name,
                    scenario=scenario,
                    horizon_or_block=horizon_or_block,
                    seasonal_period=config.seasonal_period,
                )
                y_true = bundle.test_target_values()
                naive_factor = calculate_naive_factor(
                    bundle.train_target_values(), bundle.seasonal_period
                )
                metrics = calculate_metrics(y_true, forecasts.tolist(), naive_factor)
                forecast_path = write_forecast_output(
                    bundle=bundle,
                    forecasts=forecasts,
                    output_dir=output_dir,
                    dataset=dataset_name,
                    scenario=scenario,
                    model_name=model_name,
                )
                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "scenario": scenario,
                    "horizon_or_block": horizon_or_block,
                    "forecast_length": len(forecasts),
                    "forecast_output_file": to_repo_relative_or_absolute(forecast_path),
                    **metrics,
                })
                print(f"    {model_name}: done")

    if not rows:
        raise ValueError("No Part 0 rows produced.")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)[RESULT_COLUMNS]
    output_path = output_dir / "baseline-traditional-results.csv"
    if args.append and output_path.exists():
        results = merge_results(pd.read_csv(output_path), results)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nResults saved to: {output_path}")
    print(results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the summary CSV and forecast-outputs subdirectory.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help="Comma-separated scenario names.",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_NAMES),
        help="Comma-separated model names.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Merge with an existing summary file, replacing duplicate dataset/scenario/model rows.",
    )
    return parser.parse_args()


def parse_selection(raw: str, allowed: tuple[str, ...], label: str) -> list[str]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unknown {label}(s): {invalid}. Valid values: {list(allowed)}")
    return values


def merge_results(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    existing = existing.reindex(columns=RESULT_COLUMNS)
    new = new.reindex(columns=RESULT_COLUMNS)
    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=RESULT_KEY_COLUMNS, keep="last")
    scenario_order = {scenario: idx for idx, scenario in enumerate(SCENARIOS)}
    model_order = {model: idx for idx, model in enumerate(MODEL_NAMES)}
    merged["_scenario_order"] = merged["scenario"].map(scenario_order).fillna(len(SCENARIOS))
    merged["_model_order"] = merged["model"].map(model_order).fillna(len(MODEL_NAMES))
    merged = merged.sort_values(
        ["dataset", "_scenario_order", "_model_order", "model"],
        kind="stable",
    )
    return merged.drop(columns=["_scenario_order", "_model_order"]).reset_index(drop=True)


def forecast_for_scenario(
    bundle: DatasetBundle,
    model_name: str,
    scenario: str,
    horizon_or_block: int,
    seasonal_period: int,
) -> np.ndarray:
    train = bundle.train_target_values()
    test = bundle.test_target_values()
    if scenario == "S1":
        return forecast_static(train, len(test), model_name, seasonal_period)
    if scenario == "S2":
        return forecast_rolling(train, test, model_name, seasonal_period, horizon=1)
    if scenario == "S3":
        return forecast_rolling(train, test, model_name, seasonal_period, horizon=horizon_or_block)
    if scenario == "S4":
        return forecast_blockwise(train, test, model_name, seasonal_period, block_size=horizon_or_block)
    raise ValueError(f"Unsupported scenario: {scenario}")


def forecast_static(
    train: np.ndarray,
    n_periods: int,
    model_name: str,
    seasonal_period: int,
) -> np.ndarray:
    return predict_from_history(train, n_periods, model_name, seasonal_period)


def forecast_rolling(
    train: np.ndarray,
    test: np.ndarray,
    model_name: str,
    seasonal_period: int,
    horizon: int,
) -> np.ndarray:
    history = train.astype(float).tolist()
    forecasts: list[float] = []
    for actual in test:
        step_forecast = predict_from_history(
            np.asarray(history, dtype=float),
            max(int(horizon), 1),
            model_name,
            seasonal_period,
        )
        forecasts.append(float(step_forecast[0]))
        history.append(float(actual))
    return np.asarray(forecasts, dtype=float)


def forecast_blockwise(
    train: np.ndarray,
    test: np.ndarray,
    model_name: str,
    seasonal_period: int,
    block_size: int,
) -> np.ndarray:
    history = train.astype(float).tolist()
    forecasts: list[float] = []
    for start in range(0, len(test), block_size):
        stop = min(start + block_size, len(test))
        block_actual = test[start:stop]
        block_forecast = predict_from_history(
            np.asarray(history, dtype=float),
            len(block_actual),
            model_name,
            seasonal_period,
        )
        forecasts.extend(float(value) for value in block_forecast)
        history.extend(float(value) for value in block_actual)
    return np.asarray(forecasts, dtype=float)


def predict_from_history(
    history: np.ndarray,
    n_periods: int,
    model_name: str,
    seasonal_period: int,
) -> np.ndarray:
    predictors: dict[str, Callable[[np.ndarray, int, int], np.ndarray]] = {
        "Naive (Flat)": flat_naive_predict,
        "Naive (Seasonal)": seasonal_naive_predict,
        "Auto-ARIMA": auto_arima_predict,
    }
    try:
        predictor = predictors[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc
    return predictor(history, n_periods, seasonal_period)


def flat_naive_predict(history: np.ndarray, n_periods: int, _: int) -> np.ndarray:
    return np.full(n_periods, float(history[-1]), dtype=float)


def seasonal_naive_predict(history: np.ndarray, n_periods: int, seasonal_period: int) -> np.ndarray:
    if len(history) < seasonal_period:
        return flat_naive_predict(history, n_periods, seasonal_period)
    extended = history.astype(float).tolist()
    forecasts: list[float] = []
    for _ in range(n_periods):
        value = float(extended[-seasonal_period])
        forecasts.append(value)
        extended.append(value)
    return np.asarray(forecasts, dtype=float)


def auto_arima_predict(history: np.ndarray, n_periods: int, seasonal_period: int) -> np.ndarray:
    train = history[-ARIMA_TRAIN_LIMIT:] if len(history) > ARIMA_TRAIN_LIMIT else history
    use_seasonal = seasonal_period > 1 and seasonal_period <= 24 and len(train) < 3000
    model = pm.auto_arima(
        train,
        seasonal=use_seasonal,
        m=seasonal_period if use_seasonal else 1,
        stepwise=True,
        max_p=3,
        max_q=3,
        max_P=1,
        max_Q=1,
        max_d=2,
        max_D=1,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
    )
    return np.asarray(model.predict(n_periods=n_periods), dtype=float)


def write_forecast_output(
    bundle: DatasetBundle,
    forecasts: np.ndarray,
    output_dir: Path,
    dataset: str,
    scenario: str,
    model_name: str,
) -> Path:
    forecast_dir = output_dir / "forecast-outputs"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    path = forecast_dir / f"{slugify(dataset)}__{scenario}__{slugify(model_name)}.csv"
    out = bundle.test_df.copy()
    out["actual"] = bundle.test_target_values()
    out["forecast"] = forecasts
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def to_repo_relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
