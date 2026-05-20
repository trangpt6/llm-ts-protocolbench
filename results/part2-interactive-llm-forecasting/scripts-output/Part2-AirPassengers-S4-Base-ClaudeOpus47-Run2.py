import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
df = df.set_index('Month')
series = df['Passengers'].astype(float)
series.index.freq = 'MS'

n = len(series)
train_size = int(0.8 * n)
train = series.iloc[:train_size]
test = series.iloc[train_size:]

block_size = 12
horizon = 12

forecasts = []
history = train.copy()

i = 0
while i < len(test):
    steps = min(horizon, len(test) - i)
    model = ExponentialSmoothing(
        history,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False,
    )
    fit = model.fit()
    fc = fit.forecast(steps)
    forecasts.extend([float(v) for v in fc.values])
    new_block = test.iloc[i:i+steps]
    history = pd.concat([history, new_block])
    i += steps

forecasts = [float(x) for x in forecasts]
print(forecasts)