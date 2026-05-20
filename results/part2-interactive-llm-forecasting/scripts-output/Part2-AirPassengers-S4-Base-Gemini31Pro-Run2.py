import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Train/test split
train_size = 115
test_size = len(y) - train_size
y_train = list(y[:train_size])
y_test = list(y[train_size:])

# Forecasting setup
block_size = 12
forecasts = []

# Block-wise rolling update
for i in range(0, test_size, block_size):
    # Determine how many steps to forecast in this block
    steps_to_forecast = min(block_size, test_size - i)
    
    # Train model
    model = ExponentialSmoothing(
        y_train,
        trend='add',
        damped_trend=False,
        seasonal='mul',
        seasonal_periods=12
    )
    
    # Fit model
    fit_model = model.fit()
    
    # Forecast
    block_forecast = fit_model.forecast(steps_to_forecast)
    forecasts.extend(block_forecast.tolist())
    
    # Update training data with ground truth
    y_train.extend(y_test[i:i+steps_to_forecast])

# Print final forecasts
print(forecasts)