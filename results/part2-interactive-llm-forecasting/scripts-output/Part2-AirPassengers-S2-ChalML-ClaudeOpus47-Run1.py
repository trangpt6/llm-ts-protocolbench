import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df['Passengers'].values.astype(float)

n = len(values)
train_size = int(0.8 * n)
lags = 12

def make_features(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

forecasts = []
history = list(values[:train_size])

for i in range(train_size, n):
    arr = np.array(history)
    X_train, y_train = make_features(arr, lags)
    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    last_window = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(last_window)[0]
    forecasts.append(float(pred))
    history.append(float(values[i]))

print(forecasts)