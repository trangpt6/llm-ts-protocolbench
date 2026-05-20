import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

y = df['% WEIGHTED ILI'].astype(float).copy()
y[y == 0.0] = np.nan
y = y.interpolate(method='linear', limit_direction='both')

n = len(y)
train_size = 1040
test_size = n - train_size

order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

history = list(y.iloc[:train_size].values.astype(float))
test_values = y.iloc[train_size:].values.astype(float)
forecasts = []
prev_params = None

for i in range(test_size):
    endog = np.asarray(history, dtype=float)
    model = SARIMAX(endog, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    try:
        if prev_params is not None:
            res = model.fit(start_params=prev_params, disp=False, maxiter=50)
        else:
            res = model.fit(disp=False, maxiter=200)
        prev_params = res.params
    except Exception:
        try:
            res = model.fit(disp=False, maxiter=100)
            prev_params = res.params
        except Exception:
            forecasts.append(float(history[-1]))
            history.append(float(test_values[i]))
            continue
    fc = res.forecast(steps=1)
    forecasts.append(float(np.asarray(fc).ravel()[0]))
    history.append(float(test_values[i]))

print(forecasts)