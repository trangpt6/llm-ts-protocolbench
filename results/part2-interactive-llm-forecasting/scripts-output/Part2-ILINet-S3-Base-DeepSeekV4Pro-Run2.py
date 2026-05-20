import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Read raw CSV
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])

# Identify rows from 2009-10-11 onward and swap the misaligned columns
cutoff = pd.Timestamp('2009-10-11')
mask = df['DATE'] >= cutoff
if mask.any():
    # Store original AGE 25-49 values (which are actually AGE 25-64 after swap)
    temp = df.loc[mask, 'AGE 25-49'].copy()
    df.loc[mask, 'AGE 25-49'] = df.loc[mask, 'AGE 25-64']
    df.loc[mask, 'AGE 25-64'] = temp

# Drop entirely empty column AGE 25-49
df.drop(columns=['AGE 25-49'], inplace=True)

# Remove rows where all values (including target) are zero
# But exclude DATE column from zero-check
zero_mask = (df.drop(columns=['DATE']) == 0).all(axis=1)
df = df[~zero_mask].copy()

# Insert missing date 2002-01-06 with NaN values
missing_date = pd.Timestamp('2002-01-06')
if missing_date not in df['DATE'].values:
    new_row = pd.DataFrame([[missing_date] + [np.nan]*(df.shape[1]-1)], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    df.sort_values('DATE', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Set DATE as index
df.set_index('DATE', inplace=True)

# Linear interpolation for the missing row (and any other NaN if present)
df.interpolate(method='linear', inplace=True)

# Ensure exact weekly frequency (Sundays)
# The data is already weekly; set frequency for statsmodels
df = df.asfreq('W-SUN')

# Target column
target = '% WEIGHTED ILI'

# Chronological 80/20 split
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train_series = df[target].iloc[:train_size]
test_series = df[target].iloc[train_size:]

# Fixed hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)
trend = 'c'

# Rolling multi-step forecast
forecasts = []
current_series = train_series.copy()

for i in range(len(test_series)):
    # Fit SARIMA on current_series
    model = SARIMAX(current_series,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=50, method='powell')
    # Forecast next 4 steps
    fcast = fitted.forecast(steps=4)
    # Take the first forecast as prediction for this test point
    forecasts.append(fcast[0])
    # Add the true value at this test point to current_series for next iteration
    true_val = test_series.iloc[i]
    current_series = pd.concat([current_series, pd.Series([true_val], index=[test_series.index[i]])])
    # Shift index to maintain weekly frequency
    current_series = current_series.asfreq('W-SUN')

# Convert forecasts to list
forecasts = forecasts

print(forecasts)