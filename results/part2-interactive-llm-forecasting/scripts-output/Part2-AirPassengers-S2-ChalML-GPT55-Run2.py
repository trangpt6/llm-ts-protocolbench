import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')

target = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
lags = 12

train_values = target[:train_size].tolist()
test_values = target[train_size:train_size + test_size].tolist()

forecasts = []
history = train_values.copy()

def make_lag_data(values, lags):
    X = []
    y = []
    for i in range(lags, len(values)):
        X.append(values[i - lags:i])
        y.append(values[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)

for i in range(test_size):
    X_train, y_train = make_lag_data(history, lags)
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    X_pred = np.array(history[-lags:], dtype=float).reshape(1, -1)
    pred = float(model.predict(X_pred)[0])
    forecasts.append(pred)
    history.append(float(test_values[i]))

print(forecasts)