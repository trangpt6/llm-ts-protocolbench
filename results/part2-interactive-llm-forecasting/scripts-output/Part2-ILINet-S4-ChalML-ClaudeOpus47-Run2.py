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
y = df['% WEIGHTED ILI'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

LAGS = 52
HORIZON = 52
BLOCK = 52

def build_xy(series, lags, horizon):
    m = len(series)
    X_rows = []
    Y_rows = []
    for t in range(lags, m - horizon + 1):
        X_rows.append(series[t - lags:t])
        Y_rows.append(series[t:t + horizon])
    return np.asarray(X_rows), np.asarray(Y_rows)

def fit_and_forecast(history, steps):
    X, Y = build_xy(history, LAGS, HORIZON)
    x_last = np.asarray(history[-LAGS:]).reshape(1, -1)
    preds = []
    for h in range(steps):
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X, Y[:, h])
        preds.append(float(model.predict(x_last)[0]))
    return preds

history = list(y[:train_size])
test = list(y[train_size:])

forecasts = []
i = 0
while i < test_size:
    steps = min(BLOCK, test_size - i)
    block_preds = fit_and_forecast(np.asarray(history, dtype=float), steps)
    forecasts.extend(block_preds)
    history.extend(test[i:i + steps])
    i += steps

print(forecasts)