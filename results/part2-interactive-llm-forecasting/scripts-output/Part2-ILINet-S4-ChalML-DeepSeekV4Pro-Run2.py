import random
import numpy as np
random.seed(42)
np.random.seed(42)

import pandas as pd
from xgboost import XGBRegressor

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing from Turn 1
# Drop entirely empty column AGE 25-49 (as decided)
if 'AGE 25-49' in df.columns:
    df.drop(columns=['AGE 25-49'], inplace=True)

# Convert DATE to datetime and sort chronologically
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract target series
target_col = '% WEIGHTED ILI'
series = df[target_col].values

# Chronological train/test split (80/20) as defined in Turn 0
n_total = len(series)
n_train = int(0.8 * n_total)
train_series = series[:n_train]
test_series = series[n_train:]

# Fixed hyperparameters (excluding multi_step_strategy which is a method descriptor)
lags = 52
n_estimators = 300
max_depth = 5
learning_rate = 0.05
block_size = 52
horizon = 52

# Prepare for block-wise rolling forecast with direct multi-step strategy
history = list(train_series)
forecasts = []

# Predict block by block
for block_start in range(0, len(test_series), block_size):
    block_end = min(block_start + block_size, len(test_series))
    block_len = block_end - block_start

    # Train direct models for each horizon step using current history
    models = []
    for h in range(1, horizon + 1):
        # Create lagged features and shifted target
        X_list = []
        y_list = []
        for i in range(lags - 1, len(history) - h):
            X_list.append(history[i - lags + 1 : i + 1])
            y_list.append(history[i + h])
        X_train = np.array(X_list)
        y_train = np.array(y_list)

        # Train XGBoost regressor
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        models.append(model)

    # Forecast the current block using the same lag window (last lags values)
    last_lags = np.array(history[-lags:]).reshape(1, -1)
    block_forecasts = []
    for h in range(1, block_len + 1):
        pred = models[h - 1].predict(last_lags)[0]
        block_forecasts.append(pred)
    forecasts.extend(block_forecasts)

    # Append true test values of this block to history for next retraining (use of ground truth)
    true_block = test_series[block_start:block_end]
    history.extend(true_block)

# Ensure exact length
assert len(forecasts) == len(test_series)

print(forecasts)