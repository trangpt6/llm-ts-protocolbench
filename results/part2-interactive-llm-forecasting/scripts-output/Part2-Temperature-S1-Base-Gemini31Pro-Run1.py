import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
target_col = 'Daily minimum temperatures'
series = df[target_col].values

# Define split point based on Turn 0
train_size = 2920
test_size = 730
train_data = series[:train_size]

# Fixed hyperparameters from Turn 2
# Note: seasonal_order period is 365. 
# Static setup: train once, recursive forecast without ground truth updates.
model = SARIMAX(train_data,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 365),
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)

# Generate recursive forecasts for the entire test period
# In SARIMAX, out-of-sample forecast() performs recursive multi-step prediction
forecasts = model_fit.forecast(steps=test_size).tolist()

# Print only the final forecast list
print(forecasts)