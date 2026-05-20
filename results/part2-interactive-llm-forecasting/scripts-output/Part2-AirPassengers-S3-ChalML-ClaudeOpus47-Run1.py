import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

y = df['Passengers'].values.astype(float)

n = len(y)
train_size = int(0.8 * n)
lags = 12
horizon = 12

def build_xy(series, lags):
    X, Y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        Y.append(series[i])
    return np.array(X), np.array(Y)

forecasts = []
for i in range(n - train_size):
    history = y[:train_size + i].copy()
    X_tr, y_tr = build_xy(history, lags)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    window = list(history[-lags:])
    preds = []
    for h in range(horizon):
        x_in = np.array(window[-lags:]).reshape(1, -1)
        p = model.predict(x_in)[0]
        preds.append(p)
        window.append(p)
    forecasts.append(float(preds[0]))

print(forecasts)