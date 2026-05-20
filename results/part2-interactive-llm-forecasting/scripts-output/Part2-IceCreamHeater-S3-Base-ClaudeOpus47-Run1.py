import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

y = df['Ice cream'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
train = y[:train_size]
test = y[train_size:]
test_size = len(test)

H = 12
n_iters = test_size - H + 1

history = list(train)
forecasts = []

for i in range(n_iters):
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=H)
    forecasts.extend([float(v) for v in np.asarray(fc).ravel()])
    history.append(float(test[i]))

forecasts = [float(v) for v in forecasts]
print(forecasts)