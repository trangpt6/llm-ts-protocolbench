import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)
df = df.set_index("Date")
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
df = df.reindex(full_idx)
df["Daily minimum temperatures"] = df["Daily minimum temperatures"].interpolate(method="linear")
y = df["Daily minimum temperatures"].astype(float).values

n_total = len(y)
n_train = int(0.8 * n_total)
train_hist = list(y[:n_train])
test_vals = y[n_train:]

block_size = 30
horizon = 30
forecasts = []

i = 0
while i < len(test_vals):
    cur_h = min(horizon, len(test_vals) - i)
    model = SARIMAX(
        np.array(train_hist, dtype=float),
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 365),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred = res.forecast(steps=cur_h)
    forecasts.extend([float(v) for v in np.asarray(pred)])
    train_hist.extend([float(v) for v in test_vals[i:i + cur_h]])
    i += cur_h

print(forecasts)