import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']
train_size = 158
test_size = 40
block_size = 12
horizon = 12
data = df[feature_cols].values
train_data = data[:train_size]
test_data = data[train_size:train_size + test_size]
def create_direct_dataset(data, lags=12, horizon=12):
    X = []
    y = []
    for i in range(lags, len(data) - horizon + 1):
        x = data[i - lags:i].flatten()
        y_val = data[i:i + horizon, 1]
        X.append(x)
        y.append(y_val)
    return np.array(X), np.array(y)
forecasts = []
current_data = train_data.copy()
for i in range(0, test_size, block_size):
    block_end = min(i + block_size, test_size)
    X_train, y_train = create_direct_dataset(current_data, lags=12, horizon=12)
    if len(X_train) == 0:
        break
    base_model = lgb.LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    last_lags = current_data[-12:].flatten().reshape(1, -1)
    pred = model.predict(last_lags)[0]
    needed = block_end - i
    block_pred = pred[:needed]
    forecasts.extend(block_pred.tolist())
    true_block = test_data[i:block_end]
    current_data = np.vstack([current_data, true_block])
print(forecasts)