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

y = df['% WEIGHTED ILI'].astype(float).values

n_total = len(y)
n_train = int(0.8 * n_total)
n_test = n_total - n_train
H = 4

forecasts = []
prev_params = None

for i in range(n_test - H + 1):
    history = y[:n_train + i]
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = None
    try:
        if prev_params is not None:
            res = model.fit(start_params=prev_params, disp=False, maxiter=50, method='lbfgs')
        else:
            res = model.fit(disp=False, maxiter=200)
        prev_params = res.params
    except Exception:
        try:
            res = model.fit(disp=False)
            prev_params = res.params
        except Exception:
            res = None
    if res is not None:
        fc = np.asarray(res.forecast(steps=H)).ravel()
        forecasts.extend([float(v) for v in fc])
    else:
        last = float(history[-1])
        forecasts.extend([last] * H)

print(forecasts)