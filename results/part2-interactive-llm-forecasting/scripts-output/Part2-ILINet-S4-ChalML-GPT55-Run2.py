import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

y = df["% WEIGHTED ILI"].astype(float).to_numpy()

train_size = 1040
test_size = 261
lags = 52
n_estimators = 300
max_depth = 5
learning_rate = 0.05
block_size = 52
horizon = 52

train_y = y[:train_size]
test_y = y[train_size:train_size + test_size]

def make_direct_training_data(history, h, lags):
    n = len(history)
    X = []
    target = []
    for i in range(lags, n - h + 1):
        X.append(history[i - lags:i])
        target.append(history[i + h - 1])
    return np.asarray(X, dtype=float), np.asarray(target, dtype=float)

def forecast_block(history, block_len):
    x_last = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
    preds = []
    for h in range(1, block_len + 1):
        X_train, y_train = make_direct_training_data(history, h, lags)
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1
        )
        model.fit(X_train, y_train)
        preds.append(float(model.predict(x_last)[0]))
    return preds

history = train_y.copy()
forecasts = []

start = 0
while start < test_size:
    block_len = min(block_size, test_size - start, horizon)
    block_preds = forecast_block(history, block_len)
    forecasts.extend(block_preds)
    history = np.concatenate([history, test_y[start:start + block_len]])
    start += block_len

forecasts = forecasts[:test_size]
print(forecasts)