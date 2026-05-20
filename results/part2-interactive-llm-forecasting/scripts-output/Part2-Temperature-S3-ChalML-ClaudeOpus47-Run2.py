import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
y = df['Daily minimum temperatures'].astype(float).values

n_total = len(y)
n_train = int(0.8 * n_total)
n_test = n_total - n_train
lags = 7
horizon = 7

forecasts = []
for step in range(n_test):
    train_data = y[:n_train + step]
    n = len(train_data)
    n_samples = n - lags - horizon + 1
    X_train = np.zeros((n_samples, lags))
    Y_train = np.zeros((n_samples, horizon))
    for i in range(n_samples):
        X_train[i] = train_data[i:i + lags]
        for h in range(horizon):
            Y_train[i, h] = train_data[i + lags + h]
    model_h1 = LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
    model_h1.fit(X_train, Y_train[:, 0])
    x_pred = train_data[-lags:].reshape(1, -1)
    pred = float(model_h1.predict(x_pred)[0])
    forecasts.append(pred)

print(forecasts)