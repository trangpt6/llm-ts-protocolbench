import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)
y = df["Daily minimum temperatures"].astype(float).values

train_size = 2920
test_size = 730
train = y[:train_size].tolist()
test = y[train_size:train_size + test_size].tolist()

horizon = 30
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 365)

forecasts = []
history = list(train)
i = 0
while i < test_size:
    steps = min(horizon, test_size - i)
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend="n",
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.forecast(steps=steps)
    forecasts.extend([float(v) for v in fc])
    history.extend(test[i:i + steps])
    i += steps

print(forecasts)