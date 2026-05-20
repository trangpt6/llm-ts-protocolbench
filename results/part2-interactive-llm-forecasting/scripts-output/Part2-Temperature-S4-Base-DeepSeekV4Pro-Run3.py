import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False)
df = df.set_index('Date')
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].astype(float)
target_col = 'Daily minimum temperatures'

# Chronological split
total_len = len(df)
train_len = int(0.8 * total_len)
train = df.iloc[:train_len]
test = df.iloc[train_len:]

# Forecasting setup parameters
block_size = 30
horizon = 30

# Initial training data
current_train = train[target_col].copy()

# Prepare forecasts list
forecasts = []

# Iterate over test set in blocks of block_size
test_start_idx = 0
while test_start_idx < len(test):
    # Determine how many steps to forecast for this block
    remaining = len(test) - test_start_idx
    steps = min(horizon, remaining)
    
    # Fit SARIMA model on current training series
    model = SARIMAX(current_train, 
                    order=(2, 0, 2),
                    seasonal_order=(1, 1, 1, 365),
                    trend='n',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(method='lbfgs', maxiter=500, disp=False, start_params=None,
                       random_state=np.random.RandomState(42))
    
    # Forecast the next block (up to horizon)
    pred = fitted.forecast(steps=horizon)
    if steps < horizon:
        pred = pred[:steps]  # only need the first steps
    
    # Store forecasts
    forecasts.extend(list(pred))
    
    # Update with true values for this block
    test_block_actual = test[target_col].iloc[test_start_idx:test_start_idx + steps]
    current_train = pd.concat([current_train, test_block_actual])
    
    # Advance index
    test_start_idx += steps

# Output the forecast list
print(forecasts)