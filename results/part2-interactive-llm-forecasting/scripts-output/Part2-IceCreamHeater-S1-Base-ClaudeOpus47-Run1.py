import numpy as np
import random
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
df.set_index('Month', inplace=True)
df.index.freq = 'MS'

target = df['Ice cream'].astype(float)

n = len(target)
train_size = int(0.8 * n)
train = target.iloc[:train_size]
test = target.iloc[train_size:]

model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted = model.fit(disp=False)

forecast_res = fitted.get_forecast(steps=len(test))
forecasts = [float(v) for v in forecast_res.predicted_mean.values]

print(forecasts)