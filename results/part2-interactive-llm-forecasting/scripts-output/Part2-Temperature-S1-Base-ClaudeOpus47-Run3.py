import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

y = df['Daily minimum temperatures'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
train = y[:train_size]
test = y[train_size:]

model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 365),
    trend='n',
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fit = model.fit(disp=False)

fc = fit.forecast(steps=len(test))
forecasts = [float(v) for v in np.asarray(fc).ravel()]

print(forecasts)