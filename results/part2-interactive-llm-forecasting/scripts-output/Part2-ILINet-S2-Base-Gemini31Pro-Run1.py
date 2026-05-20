import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

# Preprocessing: Handle missing timestamp 2002-01-06
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_range)
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# Preprocessing: Handle missing exogenous columns
# AGE 25-49 and AGE 50-64 have missing values due to structural reporting changes
df['AGE 25-49'] = df['AGE 25-49'].fillna(0)
df['AGE 50-64'] = df['AGE 50-64'].fillna(0)

# Define split based on Turn 0
total_timesteps = len(df)
train_size = 1044
test_size = total_timesteps - train_size

target_series = df['% WEIGHTED ILI'].values
train_data = target_series[:train_size]
test_data = target_series[train_size:]

# Fixed Setup: SARIMA retrained at every step for one-step ahead rolling forecast
history = list(train_data)
forecasts = []

order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)

for i in range(len(test_data)):
    # Retrain model at every step with all available ground truth
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend='n', enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # One-step ahead forecast
    yhat = model_fit.forecast()[0]
    forecasts.append(yhat)
    
    # Update history with ground truth for next iteration
    history.append(test_data[i])

print(forecasts)