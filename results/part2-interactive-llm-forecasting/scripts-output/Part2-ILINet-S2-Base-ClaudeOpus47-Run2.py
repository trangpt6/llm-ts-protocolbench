import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
target_col = '% WEIGHTED ILI'
df = df[['DATE', target_col]]
df = df.set_index('DATE')
df = df.asfreq('W-SUN')
if df[target_col].isna().any():
    df[target_col] = df[target_col].ffill()

series = df[target_col].astype(float).values
n_total = len(series)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

forecasts = []
history = list(series[:n_train])

for i in range(n_test):
    endog = np.array(history, dtype=float)
    try:
        model = SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                        trend=trend, enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=50, method='lbfgs')
        yhat = float(res.forecast(steps=1)[0])
    except Exception:
        yhat = float(endog[-1])
    forecasts.append(yhat)
    history.append(float(series[n_train + i]))

print(forecasts)