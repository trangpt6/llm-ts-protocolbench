import pandas as pd
import numpy as np
import random
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Ignore warnings
warnings.filterwarnings("ignore")

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Define train and test splits
train_size = 115
train = list(y[:train_size])
test = list(y[train_size:])

# Initialize flat list for forecasts
forecasts = []

# Rolling update forecasting strategy
for i in range(len(test)):
    # Use ground truth during inference
    current_history = train + test[:i]
    
    # Initialize and train model
    model = SARIMAX(current_history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='n')
    fit_model = model.fit(disp=False)
    
    # Forecast 12 steps ahead
    pred = fit_model.forecast(steps=12)
    
    # Append to flat list
    forecasts.extend(pred.tolist())

# Print final forecast list
print(forecasts)