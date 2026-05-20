import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Read raw CSV
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing: drop entirely empty columns
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True, errors='ignore')

# Parse dates and set index
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)
df.set_index('DATE', inplace=True)

# Reindex to weekly frequency (Sundays) and forward fill to handle missing weeks
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_range, method='ffill')

# Extract target series
target_col = '% WEIGHTED ILI'
y = df[target_col].values

# Chronological split: first 1050 train, rest test
train_size = 1050
train_series = y[:train_size].copy()
test_series = y[train_size:].copy()

# Fixed hyperparameters
lags = 52
n_estimators = 300
max_depth = 5
learning_rate = 0.05
block_size = 52

# Function to create lagged features for direct multi-step forecasting
def create_lag_features(series, lags):
    # Returns matrix of shape (n_samples, lags) where each row is the previous lags values
    X = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
    return np.array(X)

# List to store forecasts for entire test period
all_forecasts = []

# Current training series, will be extended after each block
current_train = train_series.copy()

# Iterate over test set in blocks of 52
test_idx = 0
while test_idx < len(test_series):
    # Determine how many steps to forecast in this block (normal 52, but less at the end)
    steps_needed = min(block_size, len(test_series) - test_idx)
    
    # For each horizon step 1..52, train a separate XGBoost model (direct strategy)
    # We'll train all 52 models before forecasting, but we only need predictions for steps_needed.
    models = []
    for h in range(1, block_size+1):
        # Build training data: lag features from current_train, target shifted by h
        if len(current_train) < lags + h:
            # Not enough data to create training samples for this horizon; edge case only if train very small
            continue
        # Extract features: for each position t where t from lags to len(current_train)-h, use lags current_train[t-lags:t]
        # Target: current_train[t+h-1] (0-indexed)
        X_list = []
        y_list = []
        for t in range(lags, len(current_train) - h + 1):
            X_list.append(current_train[t-lags:t])
            y_list.append(current_train[t+h-1])
        if len(X_list) == 0:
            # Fallback: use a naive predictor (mean of last lags) if no training data for this horizon
            models.append(None)
            continue
        X_train = np.array(X_list)
        y_train = np.array(y_list)
        
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 learning_rate=learning_rate, random_state=42)
        model.fit(X_train, y_train)
        models.append(model)
    
    # Forecast: use the last `lags` values of current_train as input
    last_lags = current_train[-lags:].reshape(1, -1)
    block_forecasts = []
    for h in range(1, steps_needed+1):
        model = models[h-1]
        if model is None:
            # If no model trained, use the mean of last lags as fallback
            pred = np.mean(last_lags)
        else:
            pred = model.predict(last_lags)[0]
        block_forecasts.append(pred)
    
    # Store forecasts
    all_forecasts.extend(block_forecasts)
    
    # Update training series with true test values of this block (ground truth)
    true_block = test_series[test_idx:test_idx+steps_needed]
    current_train = np.concatenate([current_train, true_block])
    
    test_idx += steps_needed

# Output the forecasts as a Python flat list
print(all_forecasts)