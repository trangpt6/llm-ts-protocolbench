import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
y = df['Passengers'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

H = 12
forecasts = []

for i in range(test_size):
    history = y[:train_size + i]
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    fc = fit.forecast(steps=H)
    forecasts.append(float(np.asarray(fc)[0]))

forecasts = [float(v) for v in forecasts]
print(forecasts)