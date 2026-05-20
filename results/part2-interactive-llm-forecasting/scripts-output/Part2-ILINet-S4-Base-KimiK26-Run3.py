import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import random
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target = '% WEIGHTED ILI'

# Preprocessing: mark zero rows (all numeric zero) as NaN
numeric_cols = df.columns.drop('DATE')
zero_rows = (df[numeric_cols] == 0).all(axis=1)
df.loc[zero_rows, target] = np.nan

# Impute missing target values (forward fill then backward fill)
df[target] = df[target].ffill().bfill()

total_rows = len(df)
train_size = int(0.8 * total_rows)
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Fixed SARIMA hyperparameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

block_size = 52
forecast_horizon = 52

train_series = train[target].values.astype(float)
test_series = test[target].values.astype(float)

all_forecasts = []
history = list(train_series)

# Iterate over test set in blocks
n_test = len(test_series)
i = 0
while i < n_test:
    # Train SARIMA on current history
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    # Determine steps to forecast (remaining test steps, capped at forecast_horizon)
    steps = min(forecast_horizon, n_test - i)
    forecast = fitted.forecast(steps=steps)
    all_forecasts.extend(forecast.tolist())
    # Add true values of this block to history for next iteration
    history.extend(test_series[i:i+steps].tolist())
    i += steps

print(all_forecasts)