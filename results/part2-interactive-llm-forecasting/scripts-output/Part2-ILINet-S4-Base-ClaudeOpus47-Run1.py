import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

y = df['% WEIGHTED ILI'].astype(float).ffill().values

train_size = 1040
test_size = 260
block_size = 52
horizon = 52

train = y[:train_size]
test = y[train_size:train_size + test_size]

forecasts = []
history = list(train)
n_blocks = test_size // block_size

for b in range(n_blocks):
    model = SARIMAX(
        history,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 52),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    try:
        fit = model.fit(disp=False, maxiter=200)
    except Exception:
        fit = model.fit(disp=False, method='nm', maxiter=500)
    fc = np.asarray(fit.forecast(steps=horizon)).ravel().tolist()
    forecasts.extend(fc)
    start = b * block_size
    end = start + block_size
    history.extend(list(test[start:end]))

remaining = test_size - n_blocks * block_size
if remaining > 0:
    model = SARIMAX(
        history,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 52),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    try:
        fit = model.fit(disp=False, maxiter=200)
    except Exception:
        fit = model.fit(disp=False, method='nm', maxiter=500)
    fc = np.asarray(fit.forecast(steps=remaining)).ravel().tolist()
    forecasts.extend(fc)

forecasts = [float(v) for v in forecasts[:test_size]]
print(forecasts)