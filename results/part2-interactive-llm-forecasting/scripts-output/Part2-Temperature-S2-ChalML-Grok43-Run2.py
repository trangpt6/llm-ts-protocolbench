import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values
train_size = 2920
train_target = target[:train_size]
non_zero = train_target[train_target != 0]
median_val = np.median(non_zero)
target[:train_size] = np.where(target[:train_size] == 0, median_val, target[:train_size])
history = list(target[:train_size])
forecasts = []
for step in range(730):
    X_train = []
    y_train = []
    for i in range(7, len(history)):
        X_train.append(history[i-7:i])
        y_train.append(history[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    X_pred = np.array([history[-7:]])
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    true_val = target[train_size + step]
    history.append(true_val)
print(forecasts)