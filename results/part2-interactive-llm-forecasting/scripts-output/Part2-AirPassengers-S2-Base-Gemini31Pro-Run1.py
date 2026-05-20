import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Suppress warnings to ensure clean output
warnings.filterwarnings("ignore")

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Define split
train_size = 115
train = list(y[:train_size])
test = list(y[train_size:])

# Initialize forecast list
forecasts = []

# Rolling update with retraining
for i in range(len(test)):
    # Initialize model with fixed hyperparameters
    model = ExponentialSmoothing(
        train,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    )
    
    # Fit model
    fit_model = model.fit()
    
    # Predict one step ahead
    pred = fit_model.forecast(1)[0]
    forecasts.append(float(pred))
    
    # Append true value to training set for the next step
    train.append(test[i])

# Print final forecasts
print(forecasts)