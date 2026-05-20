import pandas as pd
import numpy as np
import lightgbm as lgb
import random
import warnings

warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
y = df['Daily minimum temperatures'].values

train_size = 2920
train = y[:train_size]
test = y[train_size:]

history = list(train)
forecasts = []

for i in range(len(test)):
    X_train = []
    y_train = []
    for j in range(7, len(history)):
        X_train.append(history[j-7:j])
        y_train.append(history[j])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    X_test = np.array(history[-7:]).reshape(1, -1)
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    history.append(test[i])

print(forecasts)