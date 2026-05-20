import random
random.seed(42)
import numpy as np
np.random.seed(42)
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
drop_cols = ['AGE 25-49', 'AGE 50-64']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

y = df.set_index('DATE')['% WEIGHTED ILI']

train_size = 1044
test_size = 262

y_train = y.iloc[:train_size]

model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c', enforce_stationarity=False, enforce_invertibility=False)
results = model.fit(disp=False)

forecasts = results.forecast(steps=test_size)
forecasts_list = forecasts.tolist()

print(forecasts_list)