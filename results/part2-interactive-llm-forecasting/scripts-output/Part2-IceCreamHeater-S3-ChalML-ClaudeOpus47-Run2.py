import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.sort_values('Month').reset_index(drop=True)

target_col = 'Ice cream'
exog_col = 'Heater'
lags = 6
n_estimators = 50
max_depth = 4
horizon = 12

total = len(df)
train_size = int(0.8 * total)
test_size = total - train_size

y_full = df[target_col].values.astype(float)
x_full = df[exog_col].values.astype(float)

def build_features(y_arr, x_arr, L):
    X, y = [], []
    for t in range(L, len(y_arr)):
        feat = list(y_arr[t-L:t][::-1]) + list(x_arr[t-L:t][::-1])
        X.append(feat)
        y.append(y_arr[t])
    return np.array(X), np.array(y)

forecasts = []
for i in range(test_size):
    end = train_size + i
    y_avail = y_full[:end]
    x_avail = x_full[:end]

    X_train, y_train = build_features(y_avail, x_avail, lags)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_hist = list(y_avail)
    x_hist = list(x_avail)
    preds = []
    for h in range(horizon):
        feat = list(reversed(y_hist[-lags:])) + list(reversed(x_hist[-lags:]))
        p = float(model.predict(np.array(feat).reshape(1, -1))[0])
        preds.append(p)
        y_hist.append(p)
        x_hist.append(x_hist[-1])

    forecasts.append(preds[0])

forecasts = [float(v) for v in forecasts]
print(forecasts)