import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
series = df['Passengers'].astype(float).values

n = len(series)
train_size = int(0.8 * n)

block_size = 12
horizon = 12

forecasts = []
current_end = train_size
while current_end < n:
    history = series[:current_end]
    model = ExponentialSmoothing(
        history,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    )
    fit = model.fit()
    fc = fit.forecast(steps=horizon)
    steps_remaining = n - current_end
    steps = min(horizon, steps_remaining)
    forecasts.extend([float(v) for v in fc[:steps]])
    current_end += block_size

print(forecasts)