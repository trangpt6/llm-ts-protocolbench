import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'])
df.set_index('Month', inplace=True)
df = df.asfreq('MS')

# Chronological split: first 115 training, remaining test
train = df.iloc[:115]
test = df.iloc[115:]

# Fixed model and hyperparameters from Turn 2
model = ExponentialSmoothing(
    train['Passengers'],
    trend='add',
    seasonal='mul',
    seasonal_periods=12,
    damped_trend=False
)
fitted = model.fit()

# Extract final states and smoothing parameters
alpha = fitted.params['smoothing_level']
beta = fitted.params['smoothing_trend']
gamma = fitted.params['smoothing_seasonal']
# Initial level and trend from fitted states
level = fitted.level[-1]
trend = fitted.trend[-1]
# Seasonal components (length 12), corresponding to months 1..12
season = fitted.season[-12:].copy()

# Determine starting month index for forecasts: last training month is July (month 7)
# month numbering: 1=Jan, ..., 12=Dec
# We need the month index for the forecast period starting Aug (8)
last_train_index = train.index[-1].month  # 7 (July)
# The test index months: 8,9,10,11,12,1,2,...,12, etc.
test_months = test.index.month

forecasts = []
for i, m in enumerate(test_months):
    # Forecast one step ahead
    y_hat = (level + trend) * season[m-1]  # multiplicative
    forecasts.append(y_hat)
    # Update state using the forecast as pseudo-observation
    # Multiplicative seasonal Holt-Winters update equations
    old_level = level
    level = alpha * (y_hat / season[m-1]) + (1 - alpha) * (old_level + trend)
    trend = beta * (level - old_level) + (1 - beta) * trend
    season[m-1] = gamma * (y_hat / level) + (1 - gamma) * season[m-1]

print(forecasts)