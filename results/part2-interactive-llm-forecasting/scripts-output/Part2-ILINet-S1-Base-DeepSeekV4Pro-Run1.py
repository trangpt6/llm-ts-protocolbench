import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read CSV
df = pd.read_csv(r'../../../data/ILINet.csv')

# Drop columns with excessive missingness
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True, errors='ignore')

# Parse dates and set index
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
df.sort_index(inplace=True)

# Regularize to weekly frequency (Sunday)
full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_index)
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].astype(float)

# Chronological split dates
train_start = '1997-10-12'
train_end = '2017-12-10'
test_start = '2017-12-17'

train_series = df.loc[train_start:train_end, '% WEIGHTED ILI'].dropna()
test_index = df.loc[test_start:, '% WEIGHTED ILI'].index

# Fit SARIMA model
model = SARIMAX(train_series,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 52),
                trend='c',
                enforce_stationarity=False,
                enforce_invertibility=False)
fit = model.fit(disp=False)

# Forecast recursively over the test period (one-step ahead, no ground truth updates)
forecast = fit.forecast(steps=len(test_index))
forecast_list = forecast.tolist()

print(forecast_list)