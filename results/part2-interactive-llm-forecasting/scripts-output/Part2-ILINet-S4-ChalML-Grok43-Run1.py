import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = '% WEIGHTED ILI'
y = df[target].values
train_size = 1044
train = y[:train_size]
test = y[train_size:]
lags = 52
horizon = 52
block_size = 52
forecasts = []
history = list(train)
for start in range(0, len(test), block_size):
    end = min(start + block_size, len(test))
    block_h = end - start
    X_train = []
    Y_train = []
    for i in range(lags, len(history) - horizon + 1):
        X_train.append(history[i-lags:i])
        Y_train.append(history[i:i+horizon])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42))
    model.fit(X_train, Y_train)
    last_window = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(last_window)[0][:block_h]
    forecasts.extend(pred.tolist())
    history.extend(test[start:end])
print(forecasts)