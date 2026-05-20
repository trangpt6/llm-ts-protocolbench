import numpy as np
np.random.seed(42)

import pandas as pd
import statsmodels.api as sm

# read data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').asfreq('H')

# chronological split: first 14016 train, rest test
train = df.iloc[:14016].copy()
test = df.iloc[14016:].copy()

# constant-value days (artifacts) to replace with NaN in training set
bad_dates = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31'
]
bad_mask = train.index.normalize().isin(pd.to_datetime(bad_dates))
train.loc[bad_mask, 'OT'] = np.nan

# minimal interpolation to fill NaNs required by SARIMA
train['OT'] = train['OT'].interpolate(method='linear')

# fit SARIMA model
model = sm.tsa.SARIMAX(
    train['OT'],
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 24),
    trend='c'
)
results = model.fit(disp=False)

# recursive one-step ahead forecasts without using true test values
forecasts = []
current_results = results
for i in range(len(test)):
    pred = current_results.forecast(steps=1).iloc[0]
    forecasts.append(pred)
    current_results = current_results.append([pred], refit=False)

print(forecasts)