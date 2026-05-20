import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

values = df[['Heater', 'Ice cream']].values.astype(float)
n = len(values)
train_size = int(0.8 * n)
test_size = n - train_size
lags = 6
horizon = 12

def make_features(series_2d, lags):
    X, y = [], []
    for i in range(lags, len(series_2d)):
        X.append(series_2d[i-lags:i].flatten())
        y.append(series_2d[i])
    return np.array(X), np.array(y)

forecasts = []

for step in range(test_size):
    end = train_size + step
    history = values[:end]
    X_train, y_train = make_features(history, lags)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    last_window = history[-lags:].copy()
    preds = []
    for h in range(horizon):
        x_input = last_window.flatten().reshape(1, -1)
        y_pred = model.predict(x_input)[0]
        preds.append(y_pred)
        last_window = np.vstack([last_window[1:], y_pred])
    forecasts.append(float(preds[0][1]))

print(forecasts)