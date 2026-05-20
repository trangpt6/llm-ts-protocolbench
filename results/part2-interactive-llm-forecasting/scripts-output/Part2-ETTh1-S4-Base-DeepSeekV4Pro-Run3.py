import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')
df = df.asfreq('h')

# Mark constant-value days as NaN
constant_days = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31',
    '2018-03-31', '2018-05-31'
]
for d in constant_days:
    df.loc[d, 'OT'] = np.nan

# Interpolate linearly to fill NaN gaps (uses only past and future within series)
df['OT'] = df['OT'].interpolate(method='linear')

series = df['OT'].values
train_size = 14016
test_size = 3504

train = series[:train_size]

# Block-wise rolling forecast
block_size = 168
forecasts = []
current_train = train.copy()
start = train_size
while start < len(series):
    steps_remaining = len(series) - start
    steps_to_forecast = min(block_size, steps_remaining)

    # Fit SARIMA with fixed hyperparameters
    model = sm.tsa.SARIMAX(current_train,
                           order=(2, 1, 2),
                           seasonal_order=(1, 1, 1, 24),
                           trend='n',
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=200, method='lbfgs')

    # Forecast 168 steps (or fewer at the end), trim to needed length
    pred = fit.forecast(steps=block_size)[:steps_to_forecast]
    forecasts.extend(pred.tolist())

    # Append actual observed block (already preprocessed) to training data
    actual_block = series[start:start + steps_to_forecast]
    current_train = np.concatenate([current_train, actual_block])
    start += steps_to_forecast

# Ensure exact length
forecasts = forecasts[:test_size]
print(forecasts)