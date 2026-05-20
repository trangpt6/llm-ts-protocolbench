import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
np.random.seed(42)

# Load the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing from Turn 1: Convert Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define the target column
target_col = 'Daily minimum temperatures'

# Split Rule from Turn 0: first 80% for training
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train_set = df.iloc[:train_size][target_col]
test_set = df.iloc[train_size:][target_col]

# Fixed Model and Hyperparameters from Turn 2
# order: [1, 0, 1], seasonal_order: [1, 1, 1, 365], trend: 'n'
model = SARIMAX(train_set,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 365),
                trend='n',
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)

# Forecasting Setup from Turn 2:
# One-step ahead prediction, Static setup, Recursive forecasting
# No ground truth used during inference, No retraining
# Forecast horizon: 1 (recursive over the test set)
forecasts_result = model_fit.forecast(steps=len(test_set))

# Convert to flat list
forecasts = forecasts_result.tolist()

# Print the final forecast list
print(forecasts)