import pandas as pd
import numpy as np
import random
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
train = y[:train_size]
test_len = n - train_size

model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
    damped_trend=False,
)
fit = model.fit()

preds = fit.forecast(test_len)
forecasts = [float(v) for v in preds]
print(forecasts)