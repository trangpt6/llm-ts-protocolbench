import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values.astype(float)

n = len(y)
train_size = int(0.8 * n)
lags = 12

def make_features(series, lags):
    X, Y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        Y.append(series[i])
    return np.array(X), np.array(Y)

history = list(y[:train_size])
test = y[train_size:]
forecasts = []

for i in range(len(test)):
    arr = np.array(history)
    X_train, y_train = make_features(arr, lags)
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    last_window = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(last_window)[0]
    forecasts.append(float(pred))
    history.append(float(test[i]))

print(forecasts)