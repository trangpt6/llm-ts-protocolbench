import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
y = df['OT'].values

# Train/test split as defined
total_rows = len(y)
train_size = int(0.8 * total_rows)
y_train_full = y[:train_size]
y_test = y[train_size:]

# Horizon and lags
H = 168
lags = 168

# LightGBM parameters (ignore 'multi_step_strategy' which is a procedure, not a model param)
params = {
    'n_estimators': 300,
    'max_depth': 5,
    'learning_rate': 0.05,
    'random_state': 42,
    'verbose': -1,
    'force_col_wise': True  # avoid warning for large dataset
}

forecasts = []
# All known values so far: initial train data
known_values = list(y_train_full)

# Number of full blocks and remainder
num_full_blocks = len(y_test) // H
remainder = len(y_test) % H

for block_idx in range(num_full_blocks + (1 if remainder > 0 else 0)):
    start_test = block_idx * H
    end_test = start_test + H if block_idx < num_full_blocks else start_test + remainder
    hor = end_test - start_test
    
    # Prepare training data from known_values
    train_y = known_values.copy()
    
    # Create lagged features for training
    X_train = []
    y_train_target = []
    for i in range(lags, len(train_y)):
        X_train.append(train_y[i-lags:i])
        y_train_target.append(train_y[i])
    X_train = np.array(X_train)
    y_train_target = np.array(y_train_target)
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train_target)
    
    # Forecast horizon for this block recursively
    # Current known window: last lags of known_values
    window = known_values[-lags:].copy()
    block_preds = []
    for step in range(hor):
        pred = model.predict(window.reshape(1, -1))[0]
        block_preds.append(pred)
        # Update window: drop oldest, append prediction
        window = np.append(window[1:], pred)
    
    forecasts.extend(block_preds)
    
    # After forecasting block, add ground truth (true test values) to known_values for retraining
    true_block = y_test[start_test:end_test]
    known_values.extend(true_block)

# Ensure forecasts list length exactly matches test set length
assert len(forecasts) == len(y_test), f"Forecast length {len(forecasts)} != test length {len(y_test)}"

# Output the forecast list (Option 1 style, but we print it at the end)
print(forecasts)