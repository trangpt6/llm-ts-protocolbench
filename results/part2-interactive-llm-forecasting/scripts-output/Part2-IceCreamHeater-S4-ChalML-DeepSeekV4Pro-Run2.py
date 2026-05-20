import random
import numpy as np
random.seed(42)
np.random.seed(42)

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
# No preprocessing needed

# Chronological split
train_size = 158
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Parameters
lags = 12
horizon = 12
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']

def create_direct_features(data, lags, horizon):
    # data: DataFrame with columns feature_cols; we need to create input X and output Y.
    # X: for each row t, we take lags values of feature_cols, shape (n_samples, lags * len(feature_cols))
    # Y: for each row t, we take next 'horizon' values of target, shape (n_samples, horizon)
    n = len(data)
    samples = n - lags - horizon + 1
    if samples <= 0:
        return None, None
    X = np.zeros((samples, lags * len(feature_cols)))
    Y = np.zeros((samples, horizon))
    for i in range(lags, n - horizon + 1):
        # past window of size lags from i-lags to i-1
        window = data.iloc[i-lags:i][feature_cols].values.flatten()
        X[i-lags] = window
        # future horizon from i to i+horizon-1
        Y[i-lags] = data.iloc[i:i+horizon][target_col].values
    return X, Y

# Prepare initial training data: up to the end of train set
train_data = df.iloc[:train_size].copy()
test_data = df.iloc[train_size:].copy()

# We'll iteratively forecast blocks
forecasts = []
current_train = train_data.copy()  # holds all available data up to current time

# The forecasting process: we need to forecast test size = 40.
# We'll do block size = horizon = 12 except last block may be shorter.
remaining = len(test_data)
start_idx = 0
while remaining > 0:
    # Determine how many steps to forecast this block (up to horizon)
    block_steps = min(horizon, remaining)
    # Fit model on current_train
    X_train, Y_train = create_direct_features(current_train, lags, horizon)
    # If not enough data to create training samples, fallback (should not happen)
    if X_train is None or len(X_train) == 0:
        # Not enough history; use naive forecast? Not needed here.
        break
    model = MultiOutputRegressor(LGBMRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42))
    model.fit(X_train, Y_train)
    # Prepare input for prediction: last lags obs from current_train
    last_window = current_train.iloc[-lags:][feature_cols].values.flatten().reshape(1, -1)
    pred_horizon = model.predict(last_window)[0]  # shape (horizon,)
    # Take only block_steps predictions if less than horizon
    pred_block = pred_horizon[:block_steps]
    forecasts.extend(pred_block.tolist())
    # Get actuals for this block from test set (ground truth used for update)
    block_actuals = test.iloc[start_idx:start_idx+block_steps]
    # Append actuals to current_train for next retraining
    current_train = pd.concat([current_train, block_actuals], ignore_index=True)
    start_idx += block_steps
    remaining -= block_steps

# Print forecasts as list
print(forecasts)