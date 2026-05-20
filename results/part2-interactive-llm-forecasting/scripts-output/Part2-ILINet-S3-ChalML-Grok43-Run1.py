import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = '% WEIGHTED ILI'
y = df[target].values
train_size = 1044
train = y[:train_size]
test = y[train_size:]
forecasts = []
model = None
for i in range(len(test)):
    if i == 0:
        train_data = train
    else:
        train_data = np.concatenate([train, test[:i]])
    lags = 13
    X_train = []
    y_train = []
    for j in range(lags, len(train_data) - 3):
        X_train.append(train_data[j-lags:j])
        y_train.append(train_data[j:j+4])
    if len(X_train) > 0:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
    last_window = train_data[-lags:]
    pred = model.predict(last_window.reshape(1, -1))[0]
    forecasts.append(float(pred[0]))
print(forecasts)