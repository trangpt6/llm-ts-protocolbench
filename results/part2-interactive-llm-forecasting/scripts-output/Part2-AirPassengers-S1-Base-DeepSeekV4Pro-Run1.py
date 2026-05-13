import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Reproducibility seed (not strictly required for this model, but included as instructed)
np.random.seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# No preprocessing needed; target is 'Passengers'
y = df['Passengers']

# Chronological split: first 115 train, last 29 test
train = y.iloc[:115]
test = y.iloc[115:]  # not used for forecasting per static setup

# Define model with fixed hyperparameters
model = ExponentialSmoothing(
    train,
    trend='add',
    damped_trend=False,
    seasonal='mul',
    seasonal_periods=12
)

# Fit once on training data
model_fit = model.fit()

# Recursively forecast 29 steps ahead (one‑step predictions combined)
forecasts = model_fit.forecast(29)

# Convert to flat Python list
forecast_list = list(forecasts)

# Output the forecast list as the only print statement
print(forecast_list)