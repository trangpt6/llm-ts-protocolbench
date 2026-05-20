import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
data = df[target_col].values

# Define split based on Turn 0
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

train_data = list(data[:train_size])
test_data = data[train_size:]

# Fixed hyperparameters from Turn 2
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

# Forecasting setup from Turn 2
block_size = 12
horizon = 12
forecasts = []

# Block-wise rolling update with retraining
# Test size is 40. Blocks: 12, 12, 12, 4
for i in range(0, test_size, block_size):
    # Determine current training set
    current_train = data[:train_size + i]
    
    # Initialize and fit model
    model = SARIMAX(current_train, 
                    order=order, 
                    seasonal_order=seasonal_order, 
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Forecast next block
    # If remaining test data is less than block_size, adjust horizon
    current_horizon = min(block_size, test_size - i)
    block_forecast = model_fit.forecast(steps=current_horizon)
    forecasts.extend(block_forecast.tolist())

# Print final forecast list
print(forecasts)