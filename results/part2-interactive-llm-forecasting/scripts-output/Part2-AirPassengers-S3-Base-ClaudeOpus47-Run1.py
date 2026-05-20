import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

H = 12
n_iter = test_size - H + 1

forecasts = []
for i in range(n_iter):
    history = y[:train_size + i]
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='n',
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.forecast(steps=H)
    forecasts.extend([float(v) for v in fc])

print(forecasts)