import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
train = y[:train_size].copy()
test = y[train_size:train_size + test_size].copy()

lags = 12
forecasts = []
history = train.copy()

for i in range(len(test)):
    X_train = []
    y_train = []
    for t in range(lags, len(history)):
        X_train.append(history[t - lags:t])
        y_train.append(history[t])
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)

    X_next = history[-lags:].reshape(1, -1)
    pred = float(model.predict(X_next)[0])
    forecasts.append(pred)

    history = np.append(history, test[i])

print(forecasts)