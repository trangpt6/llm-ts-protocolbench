import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
lags = 12
horizon = 12
block_size = 12

train_y = y[:train_size]
test_y = y[train_size:train_size + test_size]

def make_direct_features(series, lags, horizon):
    X = []
    Y = []
    for i in range(lags, len(series) - horizon + 1):
        X.append(series[i - lags:i])
        Y.append(series[i:i + horizon])
    return np.asarray(X), np.asarray(Y)

history = train_y.copy()
forecasts = []

while len(forecasts) < test_size:
    X_train, Y_train = make_direct_features(history, lags, horizon)
    base_model = LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, Y_train)
    x_pred = history[-lags:].reshape(1, -1)
    block_pred = model.predict(x_pred).ravel()
    steps_needed = min(block_size, test_size - len(forecasts))
    forecasts.extend(block_pred[:steps_needed].tolist())
    start = len(forecasts) - steps_needed
    end = len(forecasts)
    history = np.concatenate([history, test_y[start:end]])

print(forecasts)