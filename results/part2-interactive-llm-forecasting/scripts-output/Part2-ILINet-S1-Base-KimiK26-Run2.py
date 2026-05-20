import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seed for reproducibility
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# parse dates and set time index
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# enforce regular weekly sunday frequency and forward fill the single missing gap
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_idx)
df = df.ffill()

# extract primary target series
y = df['% WEIGHTED ILI']

# apply fixed chronological split from Turn 0
train = y.loc[:'2017-12-24']
test = y.loc['2017-12-31':]

# fit SARIMA once on training data with fixed hyperparameters
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c')
result = model.fit(disp=False)

# generate recursive one-step ahead forecasts for the entire test set without true-value updates
forecasts = result.get_forecast(steps=len(test)).predicted_mean.tolist()

print(forecasts)