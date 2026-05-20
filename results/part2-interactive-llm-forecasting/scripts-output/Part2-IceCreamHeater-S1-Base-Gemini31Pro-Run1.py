import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = df['Ice cream'].values

# Define split point based on Turn 0
train_size = 158
test_size = 40
train_data = target[:train_size]

# Fixed hyperparameters from Turn 2
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

# Initialize and fit SARIMA model on training data
model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order, trend=trend)
model_fit = model.fit(disp=False)

# Static forecasting strategy: recursive one-step ahead without ground truth updates
# This is equivalent to a multi-step out-of-sample forecast for the test period
forecasts = model_fit.forecast(steps=test_size).tolist()

# Print the final forecast list
print(forecasts)