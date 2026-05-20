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

total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train = df.iloc[:train_size]

y_train = train["Passengers"].astype(float)

model = ExponentialSmoothing(
    y_train,
    damped_trend=False,
    seasonal="mul",
    seasonal_periods=12,
    trend="add"
)

fit = model.fit()
forecasts = fit.forecast(total_timesteps - train_size).tolist()

print(forecasts)