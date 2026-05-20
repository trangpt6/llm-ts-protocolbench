import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date')
target = df['Daily minimum temperatures'].values.astype(float)
zeros_idx = np.where(target == 0)[0]
for idx in zeros_idx:
    if idx >= 7:
        median = np.median(target[idx-7:idx])
        target[idx] = median
train_size = 2920
train = target[:train_size]
test = target[train_size:]
forecasts = []
current_train = list(train)
lags = 7
for i in range(len(test)):
    X_train = []
    y_train = []
    for j in range(lags, len(current_train)):
        X_train.append(current_train[j-lags:j])
        y_train.append(current_train[j])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    last_lags = np.array(current_train[-lags:]).reshape(1, -1)
    pred = model.predict(last_lags)[0]
    forecasts.append(pred)
    current_train.append(test[i])
print(forecasts)