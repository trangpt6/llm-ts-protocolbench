import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
y = df['Daily minimum temperatures'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

lags = 7

def make_features(series, lags):
    X, Y = [], []
    for i in range(lags, len(series)):
        X.append(series[i - lags:i])
        Y.append(series[i])
    return np.array(X), np.array(Y)

forecasts = []
history = list(y[:train_size])

for i in range(test_size):
    arr = np.array(history)
    X_train, y_train = make_features(arr, lags)
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    last_window = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(last_window)[0]
    forecasts.append(float(pred))
    history.append(float(y[train_size + i]))

print(forecasts)