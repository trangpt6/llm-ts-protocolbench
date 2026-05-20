import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target = df['% WEIGHTED ILI'].values.astype(float)
n = len(target)
train_size = int(0.8 * n)
test_size = n - train_size

lags = 52
horizon = 52
block_size = 52

def build_supervised(series, lags, horizon):
    series = np.asarray(series, dtype=float)
    n_samples = len(series) - lags - horizon + 1
    X = np.zeros((n_samples, lags))
    Y = np.zeros((n_samples, horizon))
    for i in range(n_samples):
        X[i] = series[i:i+lags]
        Y[i] = series[i+lags:i+lags+horizon]
    return X, Y

def train_models(series, lags, horizon, n_horizons):
    X, Y = build_supervised(series, lags, horizon)
    models = []
    for h in range(n_horizons):
        m = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        m.fit(X, Y[:, h])
        models.append(m)
    return models

forecasts = []
history = list(target[:train_size])
block_start = train_size
remaining = test_size

while remaining > 0:
    cur_h = min(block_size, remaining)
    models = train_models(np.array(history, dtype=float), lags, horizon, cur_h)
    last_window = np.array(history[-lags:], dtype=float).reshape(1, -1)
    for h in range(cur_h):
        pred = models[h].predict(last_window)[0]
        forecasts.append(float(pred))
    truths = target[block_start:block_start+cur_h]
    history.extend(list(truths))
    block_start += cur_h
    remaining -= cur_h

print(forecasts)