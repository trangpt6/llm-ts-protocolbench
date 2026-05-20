import pandas as pd
import numpy as np
import xgboost as xgb
import random

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)
df.reset_index(drop=True, inplace=True)

target_col = '% WEIGHTED ILI'
data = df[target_col].values

train_size = 1044
train_data = data[:train_size]
test_data = data[train_size:]

n_lags = 52
horizon = 52
n_test = len(test_data)

def create_lagged_targets(series, lags):
    # returns (samples, features) and list of target arrays for each lead 1..horizon
    X = []
    Y = {h: [] for h in range(1, horizon+1)}
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        for h in range(1, horizon+1):
            if i + h <= len(series):
                Y[h].append(series[i+h-1])
    return np.array(X), Y

def train_models(train_series):
    X, Y = create_lagged_targets(train_series, n_lags)
    models = []
    for h in range(1, horizon+1):
        if len(Y[h]) == 0:
            models.append(None)
            continue
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X[:len(Y[h])], np.array(Y[h]))
        models.append(model)
    return models

current_train = train_data.copy()
forecasts = []

block_start = 0
while block_start < n_test:
    block_end = min(block_start + horizon, n_test)
    block_len = block_end - block_start
    
    models = train_models(current_train)
    
    # features for this block: last n_lags values of current_train
    if len(current_train) >= n_lags:
        feat = current_train[-n_lags:].reshape(1, -1)
    else:
        # should not happen because initial train is long enough
        feat = np.pad(current_train, (n_lags - len(current_train), 0), 'constant')[-n_lags:].reshape(1, -1)
    
    block_preds = []
    for h in range(1, block_len+1):
        pred = models[h].predict(feat)[0]
        block_preds.append(pred)
    
    forecasts.extend(block_preds)
    
    # update current_train with true test values for this block
    current_train = np.concatenate([current_train, test_data[block_start:block_end]])
    block_start = block_end

# forecasts list should have length n_test
print(forecasts)