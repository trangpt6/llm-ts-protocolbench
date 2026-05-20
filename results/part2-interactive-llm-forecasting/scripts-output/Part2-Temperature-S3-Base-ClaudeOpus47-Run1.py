import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility seeds
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.set_index('Date')

full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df = df.reindex(full_idx)
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].ffill()

y = df['Daily minimum temperatures'].values.astype(float)

n_total = len(y)
n_train = int(0.8 * n_total)
n_test = n_total - n_train
H = 7

train = y[:n_train]
test = y[n_train:]

forecasts = []
history = list(train)
prev_params = None

n_origins = n_test - H + 1

for i in range(n_origins):
    model = SARIMAX(
        history,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 365),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    try:
        if prev_params is not None:
            fit = model.fit(disp=False, start_params=prev_params, maxiter=50)
        else:
            fit = model.fit(disp=False, maxiter=200)
        prev_params = fit.params
    except Exception:
        fit = model.fit(disp=False, maxiter=50)
        prev_params = fit.params
    fc = fit.forecast(steps=H)
    forecasts.extend([float(v) for v in fc])
    history.append(float(test[i]))

print(forecasts)