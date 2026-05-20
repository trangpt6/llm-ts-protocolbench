import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

y = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
lags = 12

y_train = y[:train_size]

X_train = []
target_train = []
for i in range(lags, train_size):
    X_train.append([y_train[i - j] for j in range(1, lags + 1)])
    target_train.append(y_train[i])

X_train = np.asarray(X_train, dtype=float)
target_train = np.asarray(target_train, dtype=float)

model = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05)
model.fit(X_train, target_train)

history = list(y_train)
forecasts = []

for _ in range(test_size):
    x = np.asarray([[history[-j] for j in range(1, lags + 1)]], dtype=float)
    pred = float(model.predict(x)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)