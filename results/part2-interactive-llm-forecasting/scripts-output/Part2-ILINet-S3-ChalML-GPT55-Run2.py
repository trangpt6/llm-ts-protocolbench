import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df = df[["DATE", "% WEIGHTED ILI"]].copy()
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

target = df["% WEIGHTED ILI"].astype(float).to_numpy()

train_size = 1040
test_size = 261
train = target[:train_size]
test = target[train_size:train_size + test_size]

lags = 13
n_estimators = 50
max_depth = 4
learning_rate = 0.1
horizon = 4

def make_lag_matrix(values, lags):
    values = np.asarray(values, dtype=float)
    X = []
    y = []
    for i in range(lags, len(values)):
        X.append(values[i - lags:i][::-1])
        y.append(values[i])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

def recursive_forecast(model, history, lags, horizon):
    hist = list(history)
    preds = []
    for _ in range(horizon):
        x = np.asarray(hist[-lags:][::-1], dtype=float).reshape(1, -1)
        pred = float(model.predict(x)[0])
        preds.append(pred)
        hist.append(pred)
    return preds

history = list(train.astype(float))
forecasts = []

for i in range(test_size):
    X_train, y_train = make_lag_matrix(history, lags)
    model = LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=1,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    step_preds = recursive_forecast(model, history, lags, horizon)
    forecasts.append(float(step_preds[0]))
    history.append(float(test[i]))

print(forecasts)