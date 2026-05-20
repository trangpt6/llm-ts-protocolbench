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
current_train = list(train)
lags = 13
params = {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'verbose': -1}
for i in range(len(test)):
    if len(current_train) >= lags:
        X_train = []
        y_train = []
        for j in range(lags, len(current_train)):
            X_train.append(current_train[j-lags:j])
            y_train.append(current_train[j])
        if len(X_train) > 0:
            model = lgb.LGBMRegressor(**params)
            model.fit(np.array(X_train), np.array(y_train))
            last_window = current_train[-lags:]
            pred = []
            for h in range(4):
                next_pred = model.predict(np.array([last_window]))[0]
                pred.append(next_pred)
                last_window = last_window[1:] + [next_pred]
            forecasts.append(pred[0])
        else:
            forecasts.append(np.mean(current_train[-4:]))
    else:
        forecasts.append(np.mean(current_train[-4:]))
    current_train.append(test[i])
print(forecasts)