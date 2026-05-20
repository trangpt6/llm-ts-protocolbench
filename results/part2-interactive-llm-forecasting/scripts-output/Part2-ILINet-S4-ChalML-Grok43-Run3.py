import pandas as pd
import numpy as np
import random
from xgboost import XGBRegressor
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
y = df['% WEIGHTED ILI'].values.astype(float)
train_size = 1043
train = y[:train_size]
test = y[train_size:]
forecasts = []
current_train = list(train)
params = {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42}
for start in range(0, len(test), 52):
    block_len = min(52, len(test) - start)
    X_train = []
    y_train = []
    for t in range(52, len(current_train)):
        X_train.append(current_train[t-52:t])
        y_train.append(current_train[t])
    if len(X_train) == 0:
        X_train = [current_train[-52:]]
        y_train = [current_train[-1]]
    model = XGBRegressor(**params)
    model.fit(np.array(X_train), np.array(y_train))
    last_window = current_train[-52:]
    block_preds = []
    temp_window = last_window.copy()
    for _ in range(block_len):
        pred = model.predict(np.array([temp_window]))[0]
        block_preds.append(pred)
        temp_window = temp_window[1:] + [pred]
    forecasts.extend(block_preds)
    current_train.extend(test[start:start+block_len])
print(forecasts)