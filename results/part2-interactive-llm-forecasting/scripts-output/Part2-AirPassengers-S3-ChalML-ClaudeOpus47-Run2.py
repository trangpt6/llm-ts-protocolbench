import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values.astype(float)

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

lags = 12
n_estimators = 50
max_depth = 4
horizon = 12

def make_lag_matrix(series, lags):
    X = []
    Y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        Y.append(series[i])
    return np.array(X), np.array(Y)

forecasts = []

for i in range(test_size):
    history = y[:train_size + i]
    X_train, y_train = make_lag_matrix(history, lags)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    last_window = list(history[-lags:])
    preds = []
    for h in range(horizon):
        x_input = np.array(last_window[-lags:]).reshape(1, -1)
        p = model.predict(x_input)[0]
        preds.append(p)
        last_window.append(p)

    forecasts.append(float(preds[0]))

print(forecasts)