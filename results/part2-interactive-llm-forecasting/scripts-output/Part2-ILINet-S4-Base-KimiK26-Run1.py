import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Read the dataset from the local CSV file (must be named 'input.csv')
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
target = df['% WEIGHTED ILI'].values

# Chronological split: first 1044 timesteps for training, remainder for test
train_size = 1044
train = target[:train_size]
test = target[train_size:]  # length 261

# Fixed hyperparameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

# Block-wise rolling forecasting
block_size = 52
horizon = 52
forecasts = []
current_train = train.copy()
test_start = 0

while test_start < len(test):
    # Determine the actual block size (remaining steps in test set)
    remaining = len(test) - test_start
    actual_horizon = min(horizon, remaining)

    # Fit SARIMA on current training data
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False, maxiter=200, method='innovations_mle')

    # Forecast the next steps (dynamic horizon for last block if needed)
    pred = results.forecast(steps=actual_horizon)
    forecasts.extend(pred.tolist())

    # Use ground truth: append true values from this block to training data
    true_block = test[test_start:test_start + actual_horizon]
    current_train = np.concatenate([current_train, true_block])

    # Move to next block
    test_start += actual_horizon

# Print exactly the forecast list as required
print(forecasts)