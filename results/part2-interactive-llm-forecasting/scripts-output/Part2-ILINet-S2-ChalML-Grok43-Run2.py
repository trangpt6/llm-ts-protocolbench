import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = '% WEIGHTED ILI'
y = df[target].values
n = len(y)
train_size = 1044
test_size = n - train_size
forecasts = []
for i in range(test_size):
    train_end = train_size + i
    train_y = y[:train_end]
    lags = 13
    X_train = []
    y_train = []
    for t in range(lags, len(train_y)):
        X_train.append(train_y[t-lags:t])
        y_train.append(train_y[t])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    last_window = train_y[-lags:]
    pred = model.predict([last_window])[0]
    forecasts.append(pred)
print(forecasts)