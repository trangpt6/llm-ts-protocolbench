import pandas as pd
import numpy as np
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Parse dates and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], dayfirst=False)
df = df.sort_values('Date').reset_index(drop=True)

# Extract target series
y = df['Daily minimum temperatures'].values

# Fixed split sizes
train_size = 2921
test_size = len(y) - train_size

# Fixed SARIMA parameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

# Prepare storage for predictions
predictions = []

# Perform rolling retraining and forecasting
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Initial training series (all data up to train_size-1)
current_series = list(y[:train_size])

for i in range(test_size):
    # Fit SARIMA on current series
    model = SARIMAX(current_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False, maxiter=500, method='nm')
    
    # Forecast 7 steps ahead
    forecast_result = model_fit.get_forecast(steps=7)
    forecast_values = forecast_result.predicted_mean
    
    # Store the 1-step ahead prediction
    predictions.append(forecast_values[0])
    
    # Append true next observation to current series for next iteration
    current_series.append(y[train_size + i])

# Output the forecast list
print(predictions)