import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
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

y_all = df[target_col].values.astype(float)
x_all = df[exog_col].values.astype(float)

def build_features(y_hist, x_hist, lags):
    n = len(y_hist)
    X = []
    Y = []
    for i in range(lags, n):
        feats = list(y_hist[i-lags:i]) + list(x_hist[i-lags:i])
        X.append(feats)
        Y.append(y_hist[i])
    return np.array(X), np.array(Y)

forecasts = []

for step in range(test_size):
    end_idx = train_size + step
    y_hist = y_all[:end_idx].copy()
    x_hist = x_all[:end_idx].copy()

    X_train, Y_train = build_features(y_hist, x_hist, lags)

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, Y_train)

    y_future = list(y_hist[-lags:])
    x_future = list(x_hist[-lags:])

    preds = []
    for h in range(horizon):
        feats = np.array(y_future[-lags:] + x_future[-lags:]).reshape(1, -1)
        pred = model.predict(feats)[0]
        preds.append(float(pred))
        y_future.append(pred)
        x_future.append(x_future[-1])

    forecasts.append(preds[0])

print(forecasts)