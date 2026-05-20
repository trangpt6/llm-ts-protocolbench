import pandas as pd
import numpy as np
import statsmodels.api as sm

# Reproducibility: set random seed
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False)
df = df.sort_values('Date').reset_index(drop=True)

# No preprocessing needed as per Turn 1

# Define target series
y = df['Daily minimum temperatures']

# Chronological split: first 2921 train, rest test (already confirmed)
train_size = 2921
train = y.iloc[:train_size]
test = y.iloc[train_size:]

# Define SARIMA model with fixed hyperparameters
model = sm.tsa.SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 365),
    trend='n'
)

# Fit the model
results = model.fit(disp=False)

# Generate recursive one-step-ahead forecasts for the entire test period (731 steps)
# The forecast() method produces dynamic forecasts using previous predictions.
forecasts = results.forecast(steps=len(test))

# Convert to list for output
forecast_list = forecasts.tolist()

print(forecast_list)