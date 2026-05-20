import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target = df['% WEIGHTED ILI'].astype(float).values

n = len(target)
train_size = int(0.8 * n)
test_size = n - train_size

H = 4
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)
trend = 'c'

forecasts = []
history = list(target[:train_size])

for i in range(test_size):
    try:
        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        fc = fitted.forecast(steps=H)
        forecasts.extend([float(x) for x in np.asarray(fc).ravel()])
    except Exception:
        last_val = float(history[-1])
        forecasts.extend([last_val] * H)
    history.append(float(target[train_size + i]))

print(forecasts)