import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# Parse time index
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
df = df.sort_index()

# Primary target
target_col = 'Ice cream'

# Train/test split: first 157 observations = train, rest = test
train_size = 157
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Fixed setup parameters
block_size = 12
forecast_horizon = 12

# Initialize with training data
history = train[target_col].copy()

# Store all forecasts
all_forecasts = []

# Block-wise rolling forecast
remaining = len(test)
block_idx = 0

while remaining > 0:
    # Determine current block size (may be smaller for last block)
    current_block_size = min(block_size, remaining)
    
    # Train SARIMA model on current history
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    fitted = model.fit(disp=False)
    
    # Forecast next steps (up to forecast_horizon, but we only use current_block_size)
    forecast_steps = min(forecast_horizon, current_block_size)
    forecast = fitted.get_forecast(steps=forecast_steps)
    pred_mean = forecast.predicted_mean.values
    
    # Append predictions
    all_forecasts.extend(pred_mean.tolist())
    
    # Update history with ground truth for the predicted period
    start_idx = block_idx * block_size
    end_idx = start_idx + forecast_steps
    true_values = test.iloc[start_idx:end_idx][target_col]
    history = pd.concat([history, true_values])
    
    # Update remaining and block index
    remaining -= forecast_steps
    block_idx += 1

# Truncate to exact test set length if needed
all_forecasts = all_forecasts[:len(test)]

print(all_forecasts)