import random
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# reproducibility
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

full_idx = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
df = df.set_index('Date').reindex(full_idx)
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='time')

y = df['Daily minimum temperatures'].astype(float).values

total = len(y)
train_size = int(0.8 * total)
test_size = total - train_size

order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)

history = list(y[:train_size])
forecasts = []

for i in range(test_size):
    model = SARIMAX(history,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend='n',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(disp=False)
    yhat = float(res.forecast(steps=1)[0])
    forecasts.append(yhat)
    history.append(float(y[train_size + i]))

print(forecasts)