import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train = df["Passengers"].iloc[:train_size]
test = df["Passengers"].iloc[train_size:]

model = ExponentialSmoothing(
    train,
    trend="add",
    damped_trend=False,
    seasonal="mul",
    seasonal_periods=12
)
fit = model.fit()
forecasts = [float(x) for x in fit.forecast(steps=len(test)).tolist()]
print(forecasts)