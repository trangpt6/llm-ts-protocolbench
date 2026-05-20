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
horizon = 12

forecasts = []

for i in range(test_size):
    available = y[:train_size + i]
    X_train = np.array([available[j - lags:j] for j in range(lags, len(available))])
    y_train = np.array([available[j] for j in range(lags, len(available))])

    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    last_window = available[-lags:].copy()
    preds = []
    for h in range(horizon):
        p = model.predict(last_window.reshape(1, -1))[0]
        preds.append(p)
        last_window = np.roll(last_window, -1)
        last_window[-1] = p

    forecasts.append(preds[0])

print(forecasts)