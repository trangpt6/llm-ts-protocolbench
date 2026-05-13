import random
random.seed(42)
import numpy as np
np.random.seed(42)
import lightgbm as lgb
import pandas as pd

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month')
series = df['Passengers'].values

# Chronological split: train 115, test 29
train = series[:115]
test = series[115:]
train_size = len(train)
test_size = len(test)

# Direct multi-step strategy: train one LightGBM model per horizon h=1..12
# Lag features: use 12 past values
lags = 12
horizon = 12
block_size = 12

# Helper: create lagged dataset for a given horizon h (1-indexed)
def create_direct_dataset(data, h):
    n = len(data)
    X, y = [], []
    # i is the index of the last lag value (0-indexed), need i >= 11 and i + h < n
    for i in range(lags-1, n - h):
        X.append(data[i - lags + 1 : i + 1])  # lags values
        y.append(data[i + h])
    return np.array(X), np.array(y)

# Initial training: train 12 models on training data only
models = {}
for h in range(1, horizon + 1):
    X_h, y_h = create_direct_dataset(train, h)
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model.fit(X_h, y_h)
    models[h] = model

# Rolling block-wise forecasting with ground truth updates
history = train.copy()
forecasts = []

# Process test in blocks of size=block_size (12), last block may be partial
test_idx = 0
while test_idx < test_size:
    # Forecast next block (up to horizon steps, but at most block_size)
    steps = min(block_size, test_size - test_idx)
    last_lags = history[-lags:]  # most recent 12 values
    block_forecast = []
    for h in range(1, steps + 1):
        pred = models[h].predict(last_lags.reshape(1, -1))[0]
        block_forecast.append(pred)
    forecasts.extend(block_forecast)

    # Update history with true values for this block (ground truth)
    true_block = test[test_idx : test_idx + steps]
    history = np.concatenate([history, true_block])

    # Retrain all 12 models on expanded history unless all test processed
    if test_idx + steps < test_size:  # more blocks remain
        for h in range(1, horizon + 1):
            X_h, y_h = create_direct_dataset(history, h)
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                verbose=-1
            )
            model.fit(X_h, y_h)
            models[h] = model

    test_idx += steps

# forecasts must be exactly 29 values
forecasts = forecasts[:test_size]
print(forecasts)