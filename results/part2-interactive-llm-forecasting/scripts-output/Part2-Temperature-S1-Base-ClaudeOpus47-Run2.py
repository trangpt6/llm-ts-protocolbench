import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

full_idx = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
series = df.set_index('Date')['Daily minimum temperatures'].reindex(full_idx)
series = series.ffill()

n = 3650
train_size = int(0.8 * n)
test_size = n - train_size

y = series.values[:n]
train = y[:train_size]
test_len = test_size

model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 365),
    trend='n',
    enforce_stationarity=False,
    enforce_invertibility=False
)
res = model.fit(disp=False)

fc = res.forecast(steps=test_len)
forecasts = [float(v) for v in np.asarray(fc).ravel().tolist()]

print(forecasts)