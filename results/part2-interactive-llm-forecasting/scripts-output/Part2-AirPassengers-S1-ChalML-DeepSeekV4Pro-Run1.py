import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'])
series = df['Passengers'].values

# Chronological split: 80% train
train_size = 115
train_series = series[:train_size]
test_size = 29

# Create lag features for training
lags = 12
X_train = []
y_train = []
for i in range(lags, len(train_series)):
    X_train.append(train_series[i-lags:i])
    y_train.append(train_series[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train LightGBM regressor with fixed hyperparameters
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# Recursive forecasting over test set
forecasts = []
last_window = list(train_series[-lags:])  # last 12 training values as initial window
for step in range(test_size):
    X_input = np.array(last_window[-lags:]).reshape(1, -1)
    pred = model.predict(X_input)[0]
    forecasts.append(pred)
    last_window.append(pred)  # update window with prediction (fifo)

print(forecasts)