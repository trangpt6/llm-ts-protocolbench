import pandas as pd
import numpy as np
import pmdarima as pm
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
# Dataset configurations based on actual project datasets
DATASETS = [
    {
        "name": "AirPassengers",
        "file_path": BASE_DIR / "data" / "AirPassengers.csv",
        "time_col": "Month",
        "target_col": "Passengers",
        "freq": "M",
        "seasonal_period": 12
    },
    {
        "name": "ETTh1",
        "file_path": BASE_DIR / "data" / "ETTh1.csv",
        "time_col": "date",
        "target_col": "OT",
        "freq": "H",
        "seasonal_period": 24  # Hourly data, daily seasonality
    },
    {
        "name": "IceCreamHeater",
        "file_path": BASE_DIR / "data" / "IceCreamHeater.csv",
        "time_col": "Month",
        "target_col": "Ice cream",
        "freq": "M",
        "seasonal_period": 12
    },
    {
        "name": "ILINet",
        "file_path": BASE_DIR / "data" / "ILINet.csv",
        "time_col": "DATE",
        "target_col": "% WEIGHTED ILI",
        "freq": "W",
        "seasonal_period": 52
    },
    {
        "name": "Temperature",
        "file_path": BASE_DIR / "data" / "Temperature.csv",
        "time_col": "Date",
        "target_col": "Daily minimum temperatures",
        "freq": "D",
        "seasonal_period": 365
    }
]


def calculate_metrics(y_true, y_pred, y_train, m=1):
    """
    m: seasonal period used for MASE denominator (naive seasonal lag-m).
    Set m=1 for non-seasonal MASE (standard lag-1 naive).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
    r2 = r2_score(y_true, y_pred)

    if len(y_train) > m:
        naive_errors = np.abs(y_train[m:] - y_train[:-m])
        naive_mae = np.mean(naive_errors)
    else:
        naive_mae = np.mean(np.abs(np.diff(y_train)))

    mase = mae / naive_mae if naive_mae != 0 else np.nan

    return {"mae": mae, "rmse": rmse, "smape": smape, "mase": mase, "r2": r2}

def seasonal_naive_predict(train_data, n_periods, m):
    preds = []
    extended = list(train_data)
    for i in range(n_periods):
        preds.append(extended[-m])
        extended.append(extended[-m])
    return np.array(preds)

ARIMA_TRAIN_LIMIT = 5000

all_results = []

for ds in DATASETS:
    print(f"Processing: {ds['name']}")
    try:
        df = pd.read_csv(ds["file_path"])
        df[ds["time_col"]] = pd.to_datetime(df[ds["time_col"]])
        df = df.sort_values(by=ds["time_col"]).reset_index(drop=True)
        df[ds["target_col"]] = df[ds["target_col"]].interpolate(method="linear").bfill().ffill()

        series = df[ds["target_col"]].values
        train_size = int(0.8 * len(series))
        train_data = series[:train_size]
        test_data = series[train_size:]
        m = ds.get("seasonal_period", 1)

        print(f"  Total: {len(series)} | Train: {len(train_data)} | Test: {len(test_data)}")

        # Flat + Seasonal Naive
        naive_flat_pred = np.full(len(test_data), fill_value=train_data[-1])
        naive_flat_metrics = calculate_metrics(test_data, naive_flat_pred, train_data, m=m)
        naive_flat_metrics.update({"dataset": ds["name"], "model": "Naive (Flat)", "scenario": "S1"})
        all_results.append(naive_flat_metrics)

        seasonal_naive_pred = seasonal_naive_predict(train_data, len(test_data), m)
        seasonal_naive_metrics = calculate_metrics(test_data, seasonal_naive_pred, train_data, m=m)
        seasonal_naive_metrics.update({"dataset": ds["name"], "model": "Naive (Seasonal)", "scenario": "S1"})
        all_results.append(seasonal_naive_metrics)

        # Auto-ARIMA with seasonal detection
        print("  Fitting Auto-ARIMA...")
        arima_train = train_data[-ARIMA_TRAIN_LIMIT:] if len(train_data) > ARIMA_TRAIN_LIMIT else train_data
        use_seasonal = (m > 1) and (m <= 24) and (len(arima_train) < 3000)
        arima_model = pm.auto_arima(
            arima_train,
            seasonal=use_seasonal,
            m=m if use_seasonal else 1,
            stepwise=True,
            max_p=3, max_q=3,
            max_P=1, max_Q=1,
            max_d=2, max_D=1,
            trace=False,
            error_action="ignore",
            suppress_warnings=True
        )
        arima_pred = arima_model.predict(n_periods=len(test_data))
        arima_metrics = calculate_metrics(test_data, arima_pred, train_data, m=m)
        arima_metrics.update({"dataset": ds["name"], "model": "Auto-ARIMA", "scenario": "S1"})
        all_results.append(arima_metrics)

        print(f"  Done: {ds['name']}")

    except Exception as e:
        print(f"  Error at {ds['name']}: {e}")

if all_results:
    results_df = pd.DataFrame(all_results)
    results_df = results_df[["dataset", "model", "scenario", "mae", "rmse", "smape", "mase", "r2"]]
    output_file = BASE_DIR / "results" / "part0-traditional-baselines" / "baseline-traditional-results.csv"
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(results_df.to_string(index=False))
