import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)
y = df["Ice cream"].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size
y_train = y[:train_size]

model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
)
res = model.fit(disp=False)

pred = res.get_forecast(steps=test_size)
forecasts = [float(v) for v in pred.predicted_mean]
print(forecasts)