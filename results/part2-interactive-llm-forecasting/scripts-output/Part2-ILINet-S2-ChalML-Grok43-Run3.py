import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
y = df['% WEIGHTED ILI'].values
train_size = 1044
history = list(y[:train_size])
forecasts = []
for i in range(len(y) - train_size):
    X_train = []
    y_train = []
    for t in range(13, len(history)):
        X_train.append(history[t-13:t])
        y_train.append(history[t])
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(np.array(X_train), np.array(y_train))
    last_lags = history[-13:]
    pred = model.predict([last_lags])[0]
    forecasts.append(float(pred))
    history.append(y[train_size + i])
print(forecasts)