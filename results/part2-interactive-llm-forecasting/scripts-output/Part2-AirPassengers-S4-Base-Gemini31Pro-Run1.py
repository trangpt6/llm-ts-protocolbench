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

# Split sizes
train_size = 115
test_size = len(y) - train_size

# Forecasting setup
block_size = 12
forecasts = []

# Block-wise rolling update with retraining
current_train_size = train_size
steps_forecasted = 0

while steps_forecasted < test_size:
    # Define current training data
    y_train = y[:current_train_size]
    
    # Initialize and fit model
    model = ExponentialSmoothing(
        y_train,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    )
    
    # Fit the model
    fitted_model = model.fit()
    
    # Determine how many steps to forecast in this block
    steps_to_forecast = min(block_size, test_size - steps_forecasted)
    
    # Forecast
    block_forecast = fitted_model.forecast(steps_to_forecast)
    forecasts.extend(block_forecast.tolist())
    
    # Update for next block
    steps_forecasted += steps_to_forecast
    current_train_size += steps_to_forecast

# Print final forecast list
print(forecasts)