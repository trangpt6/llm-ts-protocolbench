import numpy as np
import random
np.random.seed(42)
random.seed(42)

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Read the dataset
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date').asfreq('h')
series = df['OT'].copy()

# Preprocessing: replace artificial constant-value blocks with NaN
# Dates of month-end days with constant values (observed from raw data)
artificial_dates = ['2016-07-31','2016-08-31','2016-10-31','2016-12-31',
                    '2017-01-31','2017-03-31','2017-05-31','2017-07-31',
                    '2017-08-31','2017-10-31','2017-12-31','2018-01-31',
                    '2018-03-31','2018-05-31']
for d in artificial_dates:
    mask = (series.index >= d) & (series.index < pd.to_datetime(d) + pd.Timedelta(days=1))
    series.loc[mask] = np.nan

# Linearly interpolate NaN values
series = series.interpolate(method='linear')

# Split
train_size = 14016
train = series.iloc[:train_size]
test = series.iloc[train_size:]

forecasts = []
history = train.tolist()
# For each test point, retrain on all available history up to that point,
# forecast 24 steps ahead, keep the first forecast, then append true value.
for i in range(len(test)):
    # Fit SARIMA
    model = SARIMAX(history, order=(1,1,1), seasonal_order=(1,1,1,24), trend='c')
    fit = model.fit(disp=False, maxiter=200)
    # Forecast next 24 steps
    pred = fit.forecast(steps=24)
    # The first forecasted value (1-step ahead) is our prediction for test[i]
    forecasts.append(pred[0])
    # Append the true test value to history for next iteration
    history.append(test.iloc[i])

print(forecasts)