import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

y = df['% WEIGHTED ILI'].astype(float).values
n_total = len(y)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

lags = 13

def build_lag_matrix(series, lags):
    n = len(series)
    X = np.zeros((n - lags, lags))
    t = np.zeros(n - lags)
    for i in range(lags, n):
        X[i - lags] = series[i - lags:i]
        t[i - lags] = series[i]
    return X, t

forecasts = []
for step in range(n_test):
    end_idx = n_train + step
    history = y[:end_idx]
    X_train, y_train = build_lag_matrix(history, lags)
    model = LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    x_pred = y[end_idx - lags:end_idx].reshape(1, -1)
    pred = float(model.predict(x_pred)[0])
    forecasts.append(pred)

print(forecasts)