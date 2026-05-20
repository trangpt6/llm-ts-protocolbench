import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)

y = df["Daily minimum temperatures"].astype(float).values

n = len(y)
train_size = int(0.8 * n)
train = list(y[:train_size])
test = y[train_size:]

order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)
trend = "n"

forecasts = []
history = list(train)

for i in range(len(test)):
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    yhat = float(res.forecast(steps=1)[0])
    forecasts.append(yhat)
    history.append(float(test[i]))

print(forecasts)