import pandas as pd
import numpy as np
import xgboost as xgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values.astype(float)
train_size = 2920
train = target[:train_size].tolist()
test = target[train_size:].tolist()
def create_lags(data, lags=14):
    X = []
    y = []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)
forecasts = []
current_train = train.copy()
for start in range(0, len(test), 30):
    end = min(start + 30, len(test))
    block_len = end - start
    X_train, y_train = create_lags(current_train, 14)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    last_lags = current_train[-14:]
    block_forecast = []
    for _ in range(block_len):
        pred = model.predict(np.array(last_lags).reshape(1, -1))[0]
        block_forecast.append(pred)
        last_lags = last_lags[1:] + [pred]
    forecasts.extend(block_forecast)
    current_train.extend(test[start:end])
print(forecasts)