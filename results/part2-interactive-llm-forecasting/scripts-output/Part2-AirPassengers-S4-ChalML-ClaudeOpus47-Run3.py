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

def build_features(series, lags, horizon):
    arr = np.asarray(series, dtype=float)
    N = len(arr)
    X = []
    Ys = [[] for _ in range(horizon)]
    for i in range(lags, N - horizon + 1):
        X.append(arr[i-lags:i][::-1])
        for h in range(1, horizon + 1):
            Ys[h-1].append(arr[i + h - 1])
    return np.array(X), [np.array(v) for v in Ys]

current_series = y[:train_size].astype(float).tolist()
test_y = y[train_size:]

forecasts = []
step = 0
while step < test_size:
    X_train, Y_trains = build_features(current_series, lags, horizon)
    models = []
    for h in range(horizon):
        m = LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
        m.fit(X_train, Y_trains[h])
        models.append(m)

    last_window = np.array(current_series[-lags:][::-1], dtype=float).reshape(1, -1)
    block_preds = [models[h].predict(last_window)[0] for h in range(horizon)]

    take = min(block_size, test_size - step)
    forecasts.extend([float(v) for v in block_preds[:take]])

    current_series.extend(test_y[step:step + take].astype(float).tolist())
    step += take

print(forecasts)