import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.sort_values('Month').reset_index(drop=True)

y = df['Ice cream'].astype(float).values
n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

block_size = 12
horizon = 12

forecasts = []
history = list(y[:train_size])
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
    fc = np.asarray(fc).tolist()
    forecasts.extend(fc)
    true_block = list(y[train_size + i: train_size + i + steps])
    history.extend(true_block)
    i += steps

forecasts = [float(x) for x in forecasts]
print(forecasts)