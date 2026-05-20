import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Read the dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Define train size based on chronological split
train_size = 115
test_size = len(y) - train_size

# Initialize list to store forecasts
forecasts = []

# Rolling forecast with retraining at every step
for i in range(test_size):
    # Current training data including ground truth up to the current step
    train_y = y[:train_size + i]
    
    # Initialize the model with fixed hyperparameters
    model = ExponentialSmoothing(
        train_y,
        trend='add',
        damped_trend=False,
        seasonal='mul',
        seasonal_periods=12
    )
    
    # Fit the model
    fit_model = model.fit()
    
    # Forecast one step ahead
    pred = fit_model.forecast(1)[0]
    forecasts.append(pred)

# Print the final forecast list
print(forecasts)