import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
target = df['Passengers'].values

train_size = 115
train_data = target[:train_size]
test_data = target[train_size:]

lags = 12
n_estimators = 300
max_depth = 6
learning_rate = 0.05

X_train = []
y_train = []
for i in range(lags, len(train_data)):
    X_train.append(train_data[i-lags:i])
    y_train.append(train_data[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
model.fit(X_train, y_train)

forecasts = []
current_history = list(train_data[-lags:])

for _ in range(len(test_data)):
    X_test = np.array(current_history[-lags:]).reshape(1, -1)
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    current_history.append(pred)

print(forecasts)