import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# reproducibility
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)
y = df["Passengers"].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size
train = y[:train_size]

model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
    damped_trend=False,
)
fit = model.fit(optimized=True)

forecasts = list(map(float, fit.forecast(test_size)))
print(forecasts)