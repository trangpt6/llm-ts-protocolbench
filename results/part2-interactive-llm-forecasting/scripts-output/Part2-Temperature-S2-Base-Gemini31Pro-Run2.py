import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing from Turn 1: Convert Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
target_col = 'Daily minimum temperatures'

# Split from Turn 0
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train_data = df[target_col].iloc[:train_size].values.tolist()
test_data = df[target_col].iloc[train_size:].values.tolist()

# Fixed Setup from Turn 2
# Model: SARIMA
# Hyperparameters: {'order': [1, 0, 1], 'seasonal_order': [1, 1, 1, 365], 'trend': 'n'}
# Strategy: One-step ahead, rolling update, retrain at every step, use ground truth
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

forecasts = []
history = list(train_data)

# Rolling forecast loop
for i in range(len(test_data)):
    # Retrain model at every step
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # One-step ahead forecast
    yhat = model_fit.forecast(steps=1)[0]
    forecasts.append(yhat)
    
    # Update history with ground truth for next iteration
    history.append(test_data[i])

# Output final forecast list
print(forecasts)