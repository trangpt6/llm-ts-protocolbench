import pandas as pd
import numpy as np
import xgboost as xgb
import random
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values.astype(float)
train_size = 2920
train = target[:train_size]
lags = 14
X_train = []
y_train = []
for i in range(lags, len(train)):
    X_train.append(train[i-lags:i])
    y_train.append(train[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
forecasts = []
current_window = list(train[-lags:])
for _ in range(730):
    pred = model.predict(np.array([current_window]))[0]
    forecasts.append(pred)
    current_window = current_window[1:] + [pred]
print(forecasts)