import pandas as pd
import numpy as np
import xgboost as xgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target_col = 'Daily minimum temperatures'
non_zero_median = df[target_col][df[target_col] != 0].median()
df[target_col] = df[target_col].replace(0, non_zero_median)
train_size = 2920
train_series = df[target_col].iloc[:train_size].values
test_series = df[target_col].iloc[train_size:].values
def create_lags(series, lags=14):
    X = []
    y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)
lags = 14
X_train, y_train = create_lags(train_series, lags)
model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
block_size = 30
forecasts = []
current_train = list(train_series)
for start in range(0, len(test_series), block_size):
    end = min(start + block_size, len(test_series))
    block_horizon = end - start
    block_forecast = []
    temp_series = current_train[-lags:]
    for _ in range(block_horizon):
        X_pred = np.array(temp_series[-lags:]).reshape(1, -1)
        pred = model.predict(X_pred)[0]
        block_forecast.append(pred)
        temp_series.append(pred)
    forecasts.extend(block_forecast)
    true_block = test_series[start:end]
    current_train.extend(true_block)
    X_train, y_train = create_lags(np.array(current_train), lags)
    model = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
print(forecasts)