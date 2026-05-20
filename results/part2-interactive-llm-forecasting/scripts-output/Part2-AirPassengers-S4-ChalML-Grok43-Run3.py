import pandas as pd
import numpy as np
import lightgbm as lgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values.astype(float)
train_size = 115
current_train = y[:train_size].tolist()
test_length = len(y) - train_size
forecasts = []
block_size = 12
while len(forecasts) < test_length:
    models = {}
    for h in range(1, 13):
        X = []
        y_h = []
        for i in range(12, len(current_train) - h + 1):
            X.append(current_train[i-12:i])
            y_h.append(current_train[i + h - 1])
        if len(X) > 0:
            X_arr = np.array(X)
            y_h_arr = np.array(y_h)
            model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
            model.fit(X_arr, y_h_arr)
            models[h] = model
    steps_to_forecast = min(block_size, test_length - len(forecasts))
    current_block_forecast = []
    last_lags = current_train[-12:]
    for h in range(1, steps_to_forecast + 1):
        if h in models:
            features = np.array(last_lags).reshape(1, -1)
            pred = models[h].predict(features)[0]
            current_block_forecast.append(pred)
    forecasts.extend(current_block_forecast)
    start_idx = train_size + len(forecasts) - len(current_block_forecast)
    true_block = y[start_idx : start_idx + len(current_block_forecast)].tolist()
    current_train.extend(true_block)
print(forecasts)