import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')

# Preprocessing: replace constant repeated values on known frozen days with NaN
frozen_days = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31',
    '2018-03-31', '2018-05-31'
]
for day in frozen_days:
    df.loc[day, 'OT'] = np.nan

# Replace negative OT values with NaN
df.loc[df['OT'] < 0, 'OT'] = np.nan

# 80/20 split
n = len(df)
train_size = int(0.8 * n)
y_all = df['OT'].values.astype(float)  # preserve NaN

# Rolling one-step ahead forecasting
forecasts = []
for i in range(train_size, n):
    # Use all data from start up to (but not including) current test point
    train_series = y_all[:i].copy()
    # Fit SARIMA model with fixed hyperparameters
    try:
        mod = sm.tsa.SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 24),
            trend='c',
            enforce_stationarity=False,
            enforce_invertibility=False,
            missing='drop'
        )
        fit = mod.fit(disp=False, maxiter=200, method='lbfgs')
        # Forecast next step
        pred = fit.forecast(steps=1)
        forecasts.append(pred[0])
    except Exception:
        # Fallback: use simple naive forecast (last valid observation)
        last_valid = train_series[~np.isnan(train_series)]
        if len(last_valid) == 0:
            forecasts.append(np.nan)
        else:
            forecasts.append(last_valid[-1])
    # Update with true observation (even if NaN) for next retraining
    # (already in y_all)

print(forecasts)