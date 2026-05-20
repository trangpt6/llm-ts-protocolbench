import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

values = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
train_values = values[:train_size]
test_values = values[train_size:train_size + test_size]

lags = 12
horizon = 12
forecasts = []
history = list(train_values)

def make_lagged_data(series, lags):
    X = []
    y = []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        y.append(series[i])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

for i in range(test_size):
    X_train, y_train = make_lagged_data(history, lags)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    recursive_history = list(history)
    step_forecasts = []
    for _ in range(horizon):
        x_input = np.asarray(recursive_history[-lags:], dtype=float).reshape(1, -1)
        y_pred = float(model.predict(x_input)[0])
        step_forecasts.append(y_pred)
        recursive_history.append(y_pred)

    forecasts.append(step_forecasts[0])
    history.append(float(test_values[i]))

print(forecasts)