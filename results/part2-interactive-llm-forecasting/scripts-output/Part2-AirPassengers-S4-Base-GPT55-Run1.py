import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

y = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
block_size = 12
horizon = 12

train = y[:train_size]
test = y[train_size:train_size + test_size]

history = train.copy()
forecasts = []
start = 0

while start < test_size:
    steps = min(horizon, block_size, test_size - start)
    model = ExponentialSmoothing(
        history,
        trend="add",
        damped_trend=False,
        seasonal="mul",
        seasonal_periods=12
    )
    fit = model.fit(optimized=True)
    block_forecast = fit.forecast(steps)
    forecasts.extend([float(v) for v in block_forecast])
    history = np.concatenate([history, test[start:start + steps]])
    start += steps

print(forecasts)