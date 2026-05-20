import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df = df.asfreq('h')  # hourly frequency

# Preprocessing: replace constant-value blocks on month-end days with NaN
# Identify month-end days
is_month_end = df.index.to_series().dt.is_month_end
# Group by date and check if all values identical
constant_mask = df.groupby(df.index.date)['OT'].transform(lambda x: x.nunique() == 1) & is_month_end
df.loc[constant_mask, 'OT'] = np.nan

# Split point (80/20)
total = len(df)
train_size = int(0.8 * total)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Full series as numpy array for convenience
series = df['OT'].values.astype(float)  # NaN where constant blocks
test_indices = np.arange(train_size, total)

# Model hyperparameters
order = (2, 1, 2)
seasonal_order = (1, 1, 1, 24)
trend = 'n'

# Block sizes
block_size = 168
forecast_horizon = 168
num_full_blocks = 3504 // block_size  # 20
remainder = 3504 % block_size  # 144

# Rolling block-wise forecast
forecasts = []
current_train_end = train_size  # index up to which we have used as training

for blk in range(num_full_blocks):
    # Training data: all observed up to start of this block (including previously observed test blocks)
    train_series = series[:current_train_end].copy()
    # Drop NaN for fitting
    train_series_clean = train_series[~np.isnan(train_series)]
    # Fit model
    model = SARIMAX(train_series_clean, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit_res = model.fit(disp=False)
    # Forecast block of size 168
    pred = fit_res.forecast(steps=forecast_horizon)
    forecasts.extend(pred.tolist())
    # Update current_train_end to include this block (true values, even if NaN, will be excluded next fit)
    current_train_end += forecast_horizon

# Last partial block
if remainder > 0:
    train_series = series[:current_train_end].copy()
    train_series_clean = train_series[~np.isnan(train_series)]
    model = SARIMAX(train_series_clean, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    fit_res = model.fit(disp=False)
    pred = fit_res.forecast(steps=remainder)
    forecasts.extend(pred.tolist())

# Ensure exact length
forecasts = forecasts[:3504]

print(forecasts)