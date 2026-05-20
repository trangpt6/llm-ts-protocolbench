import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
np.random.seed(0)
import random
random.seed(0)

# Read data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date')
df = df.asfreq('H')

# Preprocessing: identify frozen days (all 24 hours constant) and replace with NaN
daily_std = df['OT'].resample('D').std()
frozen_dates = daily_std[daily_std == 0].index
for d in frozen_dates:
    df.loc[d.strftime('%Y-%m-%d'), 'OT'] = np.nan

# Linearly interpolate missing values
df['OT'] = df['OT'].interpolate(method='linear')

# Train/test split
train_size = 14016
train = df.iloc[:train_size]
test = df.iloc[train_size:]
test_len = len(test)

# One-step ahead rolling forecast with retraining after each step
history = train['OT'].copy().tolist()
forecasts = []

# Order and seasonal order from Turn 2
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)

for i in range(test_len):
    # Fit SARIMAX on current history
    model = SARIMAX(history,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend='c',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fit = model.fit(disp=False)
    
    # One-step forecast
    pred = fit.forecast(steps=1)[0]
    forecasts.append(pred)
    
    # Update history with true value from test set
    history.append(test['OT'].iloc[i])

print(forecasts)