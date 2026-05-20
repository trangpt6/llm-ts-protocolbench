import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target = df['% WEIGHTED ILI'].values.astype(float)

n = len(target)
train_size = int(0.8 * n)
test_size = n - train_size

LAGS = 13
H = 4

def make_lag_matrix(series, lags):
    X = []
    y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

forecasts = []

for i in range(test_size):
    history = target[:train_size + i].copy()
    X_train, y_train = make_lag_matrix(history, LAGS)
    model = LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    last_window = list(history[-LAGS:])
    preds = []
    for h in range(H):
        x = np.array(last_window[-LAGS:]).reshape(1, -1)
        p = model.predict(x)[0]
        preds.append(p)
        last_window.append(p)
    forecasts.append(float(preds[0]))

print(forecasts)