import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Apply preprocessing steps
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define target variable
y = df['Daily minimum temperatures'].values

# Chronological 80/20 train/test split
train_size = int(0.8 * len(y))
train = list(y[:train_size])
test = list(y[train_size:])

# Forecasting setup: block-wise rolling update with retraining
forecasts = []
block_size = 30

for i in range(0, len(test), block_size):
    # Determine the horizon for the current block
    current_horizon = min(block_size, len(test) - i)
    
    # Initialize and train the SARIMA model on the current training set
    model = SARIMAX(train, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n')
    fit_model = model.fit(disp=False)
    
    # Forecast the next block
    block_forecast = fit_model.forecast(steps=current_horizon)
    forecasts.extend(block_forecast.tolist())
    
    # Update the training set with the true values from the current block
    train.extend(test[i:i+current_horizon])

# Print only the final forecast list
print(forecasts)