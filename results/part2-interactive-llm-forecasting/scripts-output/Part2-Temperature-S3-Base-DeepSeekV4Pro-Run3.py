import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False)
df = df.sort_values('Date').set_index('Date')
series = df['Daily minimum temperatures']

# Chronological split (80/20, exact counts as determined in Turn 0)
train = series.iloc[:2921]
test = series.iloc[2921:]

forecasts = []
history = train.copy()

# Fixed hyperparameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

for i in range(len(test)):
    # Fit SARIMA model on current history
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        random_state=42,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    # Fit with minimal output
    fit = model.fit(disp=False)
    
    # Forecast horizon = 7 (multi-step ahead per prediction step)
    forecast_horizon = fit.forecast(steps=7)
    
    # Keep only the 1-step ahead forecast for covering the test set
    yhat = forecast_horizon.iloc[0]
    forecasts.append(yhat)
    
    # Update history with the true observed value (ground truth)
    history = pd.concat([history, pd.Series([test.iloc[i]], index=[test.index[i]])])

# Output the forecasts list exactly as required
print(forecasts)