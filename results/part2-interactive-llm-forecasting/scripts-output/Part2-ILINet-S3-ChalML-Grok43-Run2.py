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
train = y[:train_size]
test = y[train_size:]
forecasts = []
model = None
for i in range(len(test)):
    train_window = y[:train_size + i]
    X_train = []
    y_train = []
    for j in range(13, len(train_window) - 3):
        X_train.append(train_window[j-13:j])
        y_train.append(train_window[j:j+4])
    if len(X_train) > 0:
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_train, y_train)
    last_window = train_window[-13:]
    pred = model.predict([last_window])[0]
    forecasts.extend(pred.tolist())
print(forecasts)