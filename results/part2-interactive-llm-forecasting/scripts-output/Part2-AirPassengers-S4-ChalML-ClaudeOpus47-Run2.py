import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

y = df['Passengers'].values.astype(float)
n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

lags = 12
horizon = 12
block_size = 12

lgb_params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42, 'verbose': -1}

def make_features(series, lags, horizon):
    series = np.asarray(series)
    L = len(series)
    rows = []
    targets = [[] for _ in range(horizon)]
    for i in range(lags - 1, L - horizon):
        feat = series[i - lags + 1 : i + 1][::-1]
        rows.append(feat)
        for h in range(horizon):
            targets[h].append(series[i + 1 + h])
    X = np.array(rows)
    Y = [np.array(t) for t in targets]
    return X, Y

def train_models(series, lags, horizon, params):
    X, Y = make_features(series, lags, horizon)
    models = []
    for h in range(horizon):
        m = LGBMRegressor(**params)
        m.fit(X, Y[h])
        models.append(m)
    return models

def forecast_block(models, series, lags, steps):
    feat = np.asarray(series[-lags:])[::-1].reshape(1, -1)
    preds = []
    for h in range(steps):
        preds.append(float(models[h].predict(feat)[0]))
    return preds

train_series = y[:train_size].copy()
test_series = y[train_size:]

forecasts = []
remaining = test_size
pos = 0
while remaining > 0:
    steps = min(block_size, remaining)
    models = train_models(train_series, lags, horizon, lgb_params)
    block_preds = forecast_block(models, train_series, lags, steps)
    forecasts.extend(block_preds)
    train_series = np.concatenate([train_series, test_series[pos:pos+steps]])
    pos += steps
    remaining -= steps

print(forecasts)