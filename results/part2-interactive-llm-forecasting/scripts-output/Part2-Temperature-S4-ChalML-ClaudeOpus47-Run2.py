import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
y = df['Daily minimum temperatures'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
y_train = y[:train_size].copy()
y_test = y[train_size:].copy()

lags = 14
block_size = 30
horizon = 30

def make_features(series, n_lags):
    X, t = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i])
        t.append(series[i])
    return np.array(X), np.array(t)

forecasts = []
history = list(y_train)
n_test = len(y_test)
i = 0
while i < n_test:
    cur_horizon = min(horizon, n_test - i)
    X_tr, y_tr = make_features(np.array(history), lags)
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=1,
        verbosity=0
    )
    model.fit(X_tr, y_tr)
    window = list(history[-lags:])
    for _ in range(cur_horizon):
        x_input = np.array(window[-lags:]).reshape(1, -1)
        pred = float(model.predict(x_input)[0])
        forecasts.append(pred)
        window.append(pred)
    history.extend(y_test[i:i + cur_horizon].tolist())
    i += cur_horizon

print(forecasts)