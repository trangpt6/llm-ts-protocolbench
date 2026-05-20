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

y = df['% WEIGHTED ILI'].values.astype(float)

n_total = len(y)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

LAGS = 13
H = 4

def make_features(series, lags):
    X, t = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        t.append(series[i])
    return np.array(X), np.array(t)

forecasts = []
num_iterations = n_test - H + 1

for it in range(num_iterations):
    end = n_train + it
    history = y[:end]
    X_train, y_train = make_features(history, LAGS)
    model = LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    cur = list(history[-LAGS:])
    for h in range(H):
        feat = np.array(cur[-LAGS:]).reshape(1, -1)
        p = float(model.predict(feat)[0])
        forecasts.append(p)
        cur.append(p)

print(forecasts)