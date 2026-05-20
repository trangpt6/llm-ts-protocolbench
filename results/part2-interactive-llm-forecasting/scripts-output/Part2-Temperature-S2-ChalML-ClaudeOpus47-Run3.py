import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').set_index('Date')

full_idx = pd.date_range(start='1981-01-01', end='1990-12-31', freq='D')
df = df.reindex(full_idx)
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='linear')

y = df['Daily minimum temperatures'].values.astype(float)

n_total = len(y)
n_train = int(0.8 * n_total)
lags = 7

def make_features(series, lags):
    X, Y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        Y.append(series[i])
    return np.array(X), np.array(Y)

forecasts = []
history = list(y[:n_train])

for step in range(n_train, n_total):
    X_train, Y_train = make_features(np.array(history), lags)
    model = LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, Y_train)
    x_input = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(x_input)[0]
    forecasts.append(float(pred))
    history.append(float(y[step]))

print(forecasts)