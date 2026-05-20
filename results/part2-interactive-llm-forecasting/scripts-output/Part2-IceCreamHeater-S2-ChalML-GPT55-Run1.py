import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]
lags = 6
train_size = 158
test_size = 40

def make_supervised(data, end_idx):
    X = []
    y = []
    for idx in range(lags, end_idx):
        row = []
        for lag in range(1, lags + 1):
            for col in feature_cols:
                row.append(data.loc[idx - lag, col])
        X.append(row)
        y.append(data.loc[idx, target_col])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def make_prediction_row(data, idx):
    row = []
    for lag in range(1, lags + 1):
        for col in feature_cols:
            row.append(data.loc[idx - lag, col])
    return np.array(row, dtype=float).reshape(1, -1)

forecasts = []

for idx in range(train_size, train_size + test_size):
    X_train, y_train = make_supervised(df, idx)
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    X_pred = make_prediction_row(df, idx)
    pred = float(model.predict(X_pred)[0])
    forecasts.append(pred)

print(forecasts)