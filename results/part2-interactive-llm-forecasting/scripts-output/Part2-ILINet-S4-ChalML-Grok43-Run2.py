import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values
train_size = 1044
train = target[:train_size]
test = target[train_size:]
lags = 52
horizon = 52
block_size = 52
X_train = []
y_train = []
for i in range(lags, len(train) - horizon + 1):
    X_train.append(train[i-lags:i])
    y_train.append(train[i:i+horizon])
X_train = np.array(X_train)
y_train = np.array(y_train)
base_model = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)
forecasts = []
current_train = list(train)
for start in range(0, len(test), block_size):
    block_end = min(start + block_size, len(test))
    block_horizon = block_end - start
    X_pred = np.array(current_train[-lags:]).reshape(1, -1)
    pred = model.predict(X_pred)[0][:block_horizon]
    forecasts.extend(pred.tolist())
    current_train.extend(test[start:block_end])
forecasts = forecasts[:len(test)]
print(forecasts)