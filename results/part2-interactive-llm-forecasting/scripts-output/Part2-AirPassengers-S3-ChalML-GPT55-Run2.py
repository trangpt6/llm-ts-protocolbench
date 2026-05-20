import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

y = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
lags = 12
horizon = 12

forecasts = []

for i in range(test_size):
    history = y[:train_size + i]

    X_train = []
    y_train = []
    for j in range(lags, len(history)):
        X_train.append(history[j - lags:j])
        y_train.append(history[j])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=4,
        random_state=0
    )
    model.fit(X_train, y_train)

    recursive_history = list(history)
    step_forecasts = []
    for _ in range(horizon):
        x_input = np.array(recursive_history[-lags:]).reshape(1, -1)
        y_pred = float(model.predict(x_input)[0])
        step_forecasts.append(y_pred)
        recursive_history.append(y_pred)

    forecasts.append(step_forecasts[0])

print(forecasts)