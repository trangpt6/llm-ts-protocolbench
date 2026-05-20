import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import random

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date')
y = df['Daily minimum temperatures'].values

train_size = 2921
train = y[:train_size]
test = y[train_size:]

n_lags = 7
horizon = 7
lgb_params = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbose': -1
}

def create_dataset(series, n_lags, horizon):
    X, y = [], []
    for i in range(n_lags, len(series) - horizon + 1):
        X.append(series[i-n_lags:i])
        y.append(series[i:i+horizon])
    return np.array(X), np.array(y)

current_series = train.copy()
X_train, y_train = create_dataset(current_series, n_lags, horizon)
model = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params))
model.fit(X_train, y_train)

forecasts = []
for t in range(len(test)):
    last_lags = current_series[-n_lags:]
    pred_7 = model.predict(last_lags.reshape(1, -1))[0]
    forecasts.append(pred_7[0])
    true_val = test[t]
    current_series = np.append(current_series, true_val)
    X_train, y_train = create_dataset(current_series, n_lags, horizon)
    model = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_params))
    model.fit(X_train, y_train)

print(forecasts)