import pandas as pd
import numpy as np
import xgboost as xgb
import random

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)
series = df['% WEIGHTED ILI'].values

total_len = len(series)
train_size = int(0.8 * total_len)
train = series[:train_size]
test = series[train_size:]
test_len = len(test)  # 261

lags = 52
n_estimators = 300
max_depth = 5
learning_rate = 0.05

def create_lagged_data(ts, lag=52):
    X = []
    for i in range(lag, len(ts)):
        X.append(ts[i-lag:i])
    return np.array(X)

# Prepare initial training data and target for each horizon
train_series = train.copy()
models = {}

for h in range(1, 53):
    # target: y_{t+h} for t from lag to len(train)-h
    Y = train_series[lags:-h] if h < (len(train_series)-lags) else train_series[lags:]
    X_lag = create_lagged_data(train_series[:len(train_series)-h+1], lag=lags) if h <= len(train_series)-lags else create_lagged_data(train_series, lag=lags)
    # Ensure lengths match
    min_len = min(len(X_lag), len(Y))
    X_lag = X_lag[:min_len]
    Y = Y[:min_len]
    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    model.fit(X_lag, Y)
    models[h] = model

# Block-wise rolling forecast
forecasts = []
current_train = train_series.copy()
block_size = 52
start_idx = len(current_train)

while start_idx < total_len:
    remaining = total_len - start_idx
    current_block_size = min(block_size, remaining)
    # Use last 52 true values from current_train as features for all forecasts in block
    last_lags = current_train[-lags:].reshape(1, -1)  # shape (1,52)
    block_forecasts = []
    for h in range(1, current_block_size+1):
        pred = models[h].predict(last_lags)[0]
        block_forecasts.append(pred)
    forecasts.extend(block_forecasts)
    # Append true values of this block (from original series) to current_train
    true_block = series[start_idx:start_idx+current_block_size]
    current_train = np.concatenate([current_train, true_block])
    # Retrain all models on updated training set
    for h in range(1, 53):
        if h >= len(current_train) - lags:
            continue
        Y = current_train[lags:-h] if h < (len(current_train)-lags) else current_train[lags:]
        X_lag = create_lagged_data(current_train[:len(current_train)-h+1], lag=lags) if h <= len(current_train)-lags else create_lagged_data(current_train, lag=lags)
        min_len = min(len(X_lag), len(Y))
        X_lag = X_lag[:min_len]
        Y = Y[:min_len]
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        model.fit(X_lag, Y)
        models[h] = model
    start_idx += current_block_size

# Ensure length matches test set
forecasts = forecasts[:test_len]
print(forecasts)