import random
random.seed(42)
import numpy as np
np.random.seed(42)
import pandas as pd
import lightgbm as lgb

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)
y = df['Daily minimum temperatures'].values

train_size = 2920
test_size = 730
y_train = y[:train_size]
y_test = y[train_size:]

lags = 7
n_estimators = 50
max_depth = 3
learning_rate = 0.1
forecasts = []

for i in range(test_size):
    current_series = np.concatenate([y_train, y_test[:i]])
    L = len(current_series)
    X_origin = current_series[L - lags:].reshape(1, -1)
    for h in range(1, 8):
        N = L - h - lags + 1
        X_h = np.array([current_series[t:t+lags] for t in range(N)])
        y_h = current_series[lags - 1 + h:L]
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbose=-1
        )
        model.fit(X_h, y_h)
        pred = model.predict(X_origin)[0]
        forecasts.append(pred)

print(forecasts)