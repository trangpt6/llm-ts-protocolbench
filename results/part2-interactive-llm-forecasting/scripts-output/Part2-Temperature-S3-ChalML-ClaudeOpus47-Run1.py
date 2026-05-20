import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
y = df['Daily minimum temperatures'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size
lags = 7
horizon = 7

forecasts = []
for i in range(test_size):
    end = train_size + i
    series = y[:end]
    n_samples = len(series) - lags - horizon + 1
    X = np.zeros((n_samples, lags))
    Y = np.zeros((n_samples, horizon))
    for j in range(n_samples):
        X[j] = series[j:j+lags]
        Y[j] = series[j+lags:j+lags+horizon]
    last_window = series[-lags:].reshape(1, -1)
    step_preds = []
    for h in range(horizon):
        model = LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
        model.fit(X, Y[:, h])
        step_preds.append(float(model.predict(last_window)[0]))
    forecasts.append(step_preds[0])

print(forecasts)