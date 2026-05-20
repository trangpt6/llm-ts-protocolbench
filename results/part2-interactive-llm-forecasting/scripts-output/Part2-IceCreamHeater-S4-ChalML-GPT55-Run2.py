import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]

train_size = 158
test_size = 40
lags = 12
n_estimators = 200
max_depth = 5
learning_rate = 0.05
block_size = 12
horizon = 12

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:train_size + test_size].copy()

def make_direct_training_data(data, h):
    X = []
    y = []
    values = data[feature_cols].to_numpy(dtype=float)
    target = data[target_col].to_numpy(dtype=float)
    for end_idx in range(lags - 1, len(data) - h):
        row = []
        for lag in range(lags):
            row.extend(values[end_idx - lag])
        X.append(row)
        y.append(target[end_idx + h])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def make_prediction_features(data):
    values = data[feature_cols].to_numpy(dtype=float)
    end_idx = len(data) - 1
    row = []
    for lag in range(lags):
        row.extend(values[end_idx - lag])
    return np.array([row], dtype=float)

forecasts = []
history_df = train_df.copy()
test_pos = 0

while test_pos < test_size:
    current_block = min(block_size, test_size - test_pos)
    block_forecasts = []
    x_pred = make_prediction_features(history_df)
    for h in range(1, current_block + 1):
        X_train, y_train = make_direct_training_data(history_df, h)
        model = LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(x_pred)[0])
        block_forecasts.append(pred)
    forecasts.extend(block_forecasts)
    true_block = test_df.iloc[test_pos:test_pos + current_block].copy()
    history_df = pd.concat([history_df, true_block], ignore_index=True)
    test_pos += current_block

print(forecasts)