import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
lags = 12

train = y[:train_size]

X_train = []
y_train = []
for i in range(lags, len(train)):
    X_train.append([train[i - j] for j in range(1, lags + 1)])
    y_train.append(train[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

model = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
model.fit(X_train, y_train)

history = list(train)
forecasts = []

for _ in range(test_size):
    x = np.array([[history[-j] for j in range(1, lags + 1)]])
    pred = float(model.predict(x)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)