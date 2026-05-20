import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values.astype(float)
train_size = 2920
train_data = target[:train_size].copy()
test_data = target[train_size:].copy()
lags = 14
n_estimators = 200
max_depth = 5
learning_rate = 0.05
def create_lags(series, lags):
    X = []
    y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)
full_history = train_data.copy()
X_train, y_train = create_lags(full_history, lags)
model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
model.fit(X_train, y_train)
forecasts = []
for start_idx in range(0, len(test_data), 30):
    block_len = min(30, len(test_data) - start_idx)
    for _ in range(block_len):
        last_lags = full_history[-lags:].reshape(1, -1)
        pred = model.predict(last_lags)[0]
        forecasts.append(pred)
        full_history = np.append(full_history, pred)
    true_values = test_data[start_idx:start_idx + block_len]
    full_history = np.concatenate([full_history[:-block_len], true_values])
    X_train, y_train = create_lags(full_history, lags)
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
    model.fit(X_train, y_train)
print(forecasts)