import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# set seeds for reproducibility
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)
y = df["Ice cream"].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

history = list(y[:train_size])
forecasts = []

for i in range(test_size):
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    yhat = float(res.forecast(steps=1)[0])
    forecasts.append(yhat)
    history.append(float(y[train_size + i]))

print(forecasts)