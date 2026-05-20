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

target_col = '% WEIGHTED ILI'
y = df[target_col].astype(float).values

n = len(y)
train_size = int(0.8 * n)
test_size = n - train_size

order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)

forecasts = []
history = list(y[:train_size])

for i in range(test_size):
    try:
        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            trend='n',
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
        yhat = float(np.asarray(fitted.forecast(steps=1)).ravel()[0])
    except Exception:
        yhat = float(history[-1])
    forecasts.append(yhat)
    history.append(float(y[train_size + i]))

print(forecasts)