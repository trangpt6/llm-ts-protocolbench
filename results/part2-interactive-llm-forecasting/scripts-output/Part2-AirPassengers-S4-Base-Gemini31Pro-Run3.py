import pandas as pd
import numpy as np
import random
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress warnings to ensure only the forecast list is printed
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Define split and block sizes
train_size = 115
block_size = 12

forecasts = []
current_train_end = train_size

# Block-wise rolling update with retraining
while current_train_end < len(y):
    train_data = y[:current_train_end]
    
    # Initialize and fit the model
    model = ExponentialSmoothing(
        train_data,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    )
    fit_model = model.fit()
    
    # Forecast the next block
    steps_to_predict = min(block_size, len(y) - current_train_end)
    preds = fit_model.forecast(steps_to_predict)
    forecasts.extend(preds.tolist())
    
    # Update the training end for the next block
    current_train_end += block_size

# Print the final forecast list
print(forecasts)