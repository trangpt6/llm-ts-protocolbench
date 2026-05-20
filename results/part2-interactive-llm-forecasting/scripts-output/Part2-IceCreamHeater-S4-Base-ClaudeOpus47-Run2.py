import numpy as np
import pandas as pd
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)
y = df["Ice cream"].astype(float).values

n = len(y)
train_size = int(0.8 * n)
y_train = y[:train_size].tolist()
y_test = y[train_size:].tolist()

block_size = 12
horizon = 12
forecasts = []

history = list(y_train)
test_len = len(y_test)
i = 0
while i < test_len:
    steps_keep = min(horizon, test_len - i)
    model = SARIMAX(history,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    trend='c',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.forecast(steps=horizon)
    fc = np.asarray(fc).tolist()
    forecasts.extend(fc[:steps_keep])
    history.extend(y_test[i:i+steps_keep])
    i += block_size

forecasts = [float(v) for v in forecasts]
print(forecasts)