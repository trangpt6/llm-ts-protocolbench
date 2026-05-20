import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set seed for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv', parse_dates=['Month'], index_col='Month')
target = df['Ice cream'].astype(float)

# Chronological split
train_size = 158
train = target.iloc[:train_size]
test = target.iloc[train_size:]

# Fixed model hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

# Store forecasts for the test set
forecasts = []

# Rolling multi-step forecasting with ground truth update and retraining
current_train = train.copy()
for i in range(len(test)):
    # Fit SARIMA model on current training data
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=200, method='lbfgs')
    
    # Forecast next 12 steps, we only need the first step (1-step ahead)
    pred = fit.forecast(steps=12)
    forecast_val = pred[0]
    forecasts.append(forecast_val)
    
    # Append ground truth for the current test point to training data
    current_train = pd.concat([current_train, pd.Series([test.iloc[i]], index=[test.index[i]])])

# Print the final forecast list
print(forecasts)