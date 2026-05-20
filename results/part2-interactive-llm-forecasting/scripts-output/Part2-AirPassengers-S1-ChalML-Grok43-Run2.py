import pandas as pd
import numpy as np
import lightgbm as lgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values
train_size = 115
train_data = data[:train_size]
lags = 12
X_train = []
y_train = []
for i in range(lags, len(train_data)):
    X_train.append(train_data[i-lags:i])
    y_train.append(train_data[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
forecasts = []
history = list(train_data[-lags:])
for _ in range(29):
    X_pred = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    history.append(pred)
print(forecasts)