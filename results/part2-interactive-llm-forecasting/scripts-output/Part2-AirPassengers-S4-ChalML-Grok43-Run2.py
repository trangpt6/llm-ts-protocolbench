import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values
train_size = 115
train_data = data[:train_size].tolist()
test_data = data[train_size:].tolist()
params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'random_state': 42}
lags = 12
block_size = 12
forecasts = []
current_data = train_data.copy()
for start in range(0, len(test_data), block_size):
    end = min(start + block_size, len(test_data))
    current_block_size = end - start
    X_train = []
    y_train = []
    for i in range(lags, len(current_data)):
        X_train.append(current_data[i-lags:i])
        y_train.append(current_data[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    last_values = current_data[-lags:].copy()
    block_forecast = []
    for _ in range(current_block_size):
        pred = model.predict([last_values])[0]
        block_forecast.append(pred)
        last_values = last_values[1:] + [pred]
    forecasts.extend(block_forecast)
    true_block = test_data[start:end]
    current_data.extend(true_block)
print(forecasts)