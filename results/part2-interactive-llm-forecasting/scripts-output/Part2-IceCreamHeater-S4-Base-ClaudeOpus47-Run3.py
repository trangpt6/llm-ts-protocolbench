import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.sort_values('Month').reset_index(drop=True)
df.set_index('Month', inplace=True)
df.index.freq = 'MS'

target = df['Ice cream'].astype(float)

n = len(target)
train_size = int(0.8 * n)
test_size = n - train_size

train = target.iloc[:train_size]
test = target.iloc[train_size:]

block_size = 12
horizon = 12

forecasts = []
history = train.copy()

i = 0
while i < test_size:
    steps = min(horizon, test_size - i)
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=steps)
    forecasts.extend(list(np.asarray(fc).ravel()))
    history = pd.concat([history, test.iloc[i:i + steps]])
    i += steps

forecasts = [float(x) for x in forecasts]
print(forecasts)