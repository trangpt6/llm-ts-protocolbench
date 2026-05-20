import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

y = df['Passengers'].values.astype(float)
n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

LAGS = 12
HORIZON = 12
BLOCK = 12
N_EST = 200
MAX_DEPTH = 5
LR = 0.05

def build_direct_dataset(series, lags, h):
    X, t = [], []
    for i in range(lags, len(series) - h + 1):
        X.append(series[i - lags:i])
        t.append(series[i + h - 1])
    return np.array(X), np.array(t)

def train_models(series):
    models = []
    for h in range(1, HORIZON + 1):
        X, t = build_direct_dataset(series, LAGS, h)
        m = LGBMRegressor(
            n_estimators=N_EST,
            max_depth=MAX_DEPTH,
            learning_rate=LR,
            random_state=42,
            verbose=-1,
        )
        m.fit(X, t)
        models.append(m)
    return models

forecasts = []
history = list(y[:train_size])
remaining = test_size
test_values = list(y[train_size:])
idx = 0

while remaining > 0:
    models = train_models(np.array(history))
    last_window = np.array(history[-LAGS:]).reshape(1, -1)
    block_steps = min(BLOCK, remaining)
    block_preds = []
    for h in range(1, block_steps + 1):
        pred = models[h - 1].predict(last_window)[0]
        block_preds.append(float(pred))
    forecasts.extend(block_preds)
    # append true values to history (use of ground truth enabled)
    history.extend(test_values[idx:idx + block_steps])
    idx += block_steps
    remaining -= block_steps

print(forecasts)