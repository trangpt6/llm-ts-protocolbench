import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
y = df["% WEIGHTED ILI"].astype(float).to_numpy()

train_size = 1040
test_size = 261
lags = 13

train_y = y[:train_size]
test_y = y[train_size:train_size + test_size]

history = list(train_y)
forecasts = []

def make_lag_matrix(values, lags):
    X = []
    target = []
    for i in range(lags, len(values)):
        X.append(values[i - lags:i][::-1])
        target.append(values[i])
    return np.asarray(X, dtype=float), np.asarray(target, dtype=float)

for i in range(test_size):
    X_train, y_train = make_lag_matrix(np.asarray(history, dtype=float), lags)
    model = LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    x_next = np.asarray(history[-lags:][::-1], dtype=float).reshape(1, -1)
    pred = float(model.predict(x_next)[0])
    forecasts.append(pred)
    history.append(float(test_y[i]))

print(forecasts)