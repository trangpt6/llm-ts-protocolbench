import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# No preprocessing needed per Turn 1 decision
# Zero values retained as valid

# Define train/test split (exact from Turn 0)
train_size = 2920
test_size = 730

train_data = df.iloc[:train_size].copy()
test_data = df.iloc[train_size:].copy()

# Extract target series
y_train = train_data['Daily minimum temperatures'].values
y_full = df['Daily minimum temperatures'].values

# Fixed hyperparameters from Turn 2
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

# Rolling one-step ahead forecasting with retraining at each step
forecasts = []

# Initial training on train set
current_train = y_train.copy()

for i in range(test_size):
    # Fit SARIMA model on current training data
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)
    
    # One-step ahead forecast
    forecast = fitted_model.forecast(steps=1)[0]
    forecasts.append(float(forecast))
    
    # Update training data with actual observed value (ground truth update)
    actual_value = y_full[train_size + i]
    current_train = np.append(current_train, actual_value)

print(forecasts)