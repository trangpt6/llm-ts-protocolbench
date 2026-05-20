import random
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date')

# Preprocessing: replace constant-value blocks on month-end days with NaN
# Identify the last calendar day of each month
last_days = df.resample('D').last().index
last_days = last_days[last_days.is_month_end]

for day in last_days:
    mask = (df.index.normalize() == day.normalize())
    if mask.any():
        values = df.loc[mask, 'OT']
        if len(values.unique()) == 1:
            df.loc[mask, 'OT'] = np.nan

# Train/test split (80/20 chronological)
n = len(df)
train_size = int(0.8 * n)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# The historical series starts with the training data (NaNs handled by SARIMAX)
history = train['OT'].copy()

# Rolling update forecasting
forecasts = []
warnings.filterwarnings("ignore")

for t_idx in range(len(test)):
    # Fit SARIMA model on current history (ignoring NaN via Kalman filter)
    model = SARIMAX(history,
                    order=(1,1,1),
                    seasonal_order=(1,1,1,24),
                    trend='c',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=50)
    # Produce 24-step ahead forecast, take only the first step
    pred = fitted.get_forecast(steps=24)
    y_pred = pred.predicted_mean[0]
    forecasts.append(y_pred)

    # Update history with the true value for the next step
    true_val = test['OT'].iloc[t_idx]
    history = pd.concat([history, pd.Series([true_val], index=[test.index[t_idx]])])

print(forecasts)