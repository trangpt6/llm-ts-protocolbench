import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
series = df['Passengers'].values
n_total = len(series)

# Fixed split
train_size = 115
test_size = n_total - train_size  # 29
train_series = list(series[:train_size])

# Fixed hyperparameters
lags = 12
n_estimators = 200
max_depth = 5
learning_rate = 0.05
multi_step_strategy = 'direct'
block_size = 12
horizon = 12

def create_supervised_dataset(series_values, lag, forecast_horizon):
    # For a single series, create features of length lag and target shifted by forecast_horizon.
    # Features: [t-1, t-2, ..., t-lag] for each valid t.
    X_list = []
    y_list = []
    T = len(series_values)
    for i in range(lag, T - forecast_horizon + 1):
        # i is index of last known time t
        feat = series_values[i-lag:i][::-1]  # most recent first: t-1, t-2, ..., t-lag
        target = series_values[i + forecast_horizon - 1]  # y_{t+h}
        X_list.append(feat)
        y_list.append(target)
    return np.array(X_list), np.array(y_list)

# Main forecasting loop
known_series = train_series.copy()  # will extend with true test values after each block
n_test_remaining = test_size
forecasts = []
block_start_index = 0  # index relative to test set

# Function to get lag feature vector from known_series for current time point
def get_prediction_features(known_series):
    # use last 'lag' values, most recent first
    return np.array(known_series[-lag:])[::-1]

current_train_end_idx = len(known_series) - 1  # last index of train data

while n_test_remaining > 0:
    # Determine how many steps to predict in this block (min between horizon and remaining)
    steps_this_block = min(horizon, n_test_remaining)
    
    # Train 12 separate direct models for horizons 1..horizon using current known_series
    models = []
    for h in range(1, horizon+1):
        X, y = create_supervised_dataset(np.array(known_series), lag, h)
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=-1
        )
        model.fit(X, y)
        models.append(model)
    
    # Prepare feature vector for this block: the last 'lag' known values
    X_pred = get_prediction_features(known_series).reshape(1, -1)
    
    # Predict for horizons 1..steps_this_block
    block_forecasts = []
    for h_idx in range(steps_this_block):
        y_pred = models[h_idx].predict(X_pred)[0]  # h_idx corresponds to horizon = h_idx+1
        block_forecasts.append(y_pred)
    forecasts.extend(block_forecasts)
    
    # Append true test values for the predicted block (ground truth update), if still within test set
    true_block_start = block_start_index
    true_block_end = block_start_index + steps_this_block
    true_values = series[train_size + true_block_start : train_size + true_block_end].tolist()
    known_series.extend(true_values)
    
    block_start_index += steps_this_block
    n_test_remaining -= steps_this_block

# The total length of forecasts should equal test_size (29)
print(forecasts)