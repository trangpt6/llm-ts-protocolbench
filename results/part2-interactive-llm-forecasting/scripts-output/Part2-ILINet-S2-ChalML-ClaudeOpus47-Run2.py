import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target = '% WEIGHTED ILI'
y = df[target].astype(float).values

n_total = len(y)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

lags = 13
n_estimators = 50
max_depth = 4
learning_rate = 0.1

def build_lag_matrix(series, lags):
    n = len(series)
    X = np.zeros((n - lags, lags))
    yv = np.zeros(n - lags)
    for i in range(lags, n):
        X[i - lags] = series[i - lags:i]
        yv[i - lags] = series[i]
    return X, yv

forecasts = []
history = list(y[:n_train])

for step in range(n_test):
    arr = np.array(history, dtype=float)
    X_train, y_train = build_lag_matrix(arr, lags)
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    last_window = arr[-lags:].reshape(1, -1)
    pred = model.predict(last_window)[0]
    forecasts.append(float(pred))
    true_val = float(y[n_train + step])
    history.append(true_val)

print(forecasts)