import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = df['Ice cream'].values

# Define split based on Turn 0
train_size = 158
test_size = 40
train_data = target[:train_size]

# Fixed hyperparameters from Turn 2
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

# Train model once (Static setup)
model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, trend=trend)
model_fit = model.fit(disp=False)

# Forecast over the test set period (40 steps)
# Static setup: no ground truth updates, recursive forecasting
forecasts = model_fit.forecast(steps=test_size).tolist()

# Print final forecast list
print(forecasts)