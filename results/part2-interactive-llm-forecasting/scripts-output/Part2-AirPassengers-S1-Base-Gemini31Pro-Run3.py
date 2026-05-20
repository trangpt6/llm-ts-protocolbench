import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set random seed for reproducibility
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# Extract target
y = df['Passengers'].values

# Train/test split
train_size = int(0.8 * len(y))
y_train = y[:train_size]

# Initialize and fit model
model = ExponentialSmoothing(
    y_train,
    trend='add',
    seasonal='mul',
    seasonal_periods=12,
    damped_trend=False
)
fitted_model = model.fit()

# Forecast
forecasts = fitted_model.forecast(len(y) - train_size)

# Print final forecast list
print(forecasts.tolist())