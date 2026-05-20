import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the dataset
df = pd.read_csv(r'../../../data/ETTh1.csv')
target = df['OT'].values

# Define the split point based on Turn 0
train_size = 13936
test_size = len(target) - train_size

train = target[:train_size]

# Initialize and fit the SARIMA model with fixed hyperparameters from Turn 2
# order: [2, 1, 2], seasonal_order: [1, 1, 1, 24], trend: 'c'
model = SARIMAX(train, 
                order=(2, 1, 2), 
                seasonal_order=(1, 1, 1, 24), 
                trend='c',
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)

# Perform recursive forecasting for the entire test set period
# In a static setup without ground truth updates, the forecast method 
# automatically uses previous predictions for subsequent steps.
forecasts = model_fit.forecast(steps=test_size).tolist()

# Print the final forecast list
print(forecasts)