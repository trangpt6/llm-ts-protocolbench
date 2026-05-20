import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility seeds
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
y = df['Ice cream'].astype(float).values

total = len(y)
train_size = int(0.8 * total)
test_size = total - train_size

H = 12
n_iter = test_size - H + 1

forecasts = []
for i in range(n_iter):
    history = y[: train_size + i]
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    fc = fit.forecast(steps=H)
    forecasts.extend([float(v) for v in np.asarray(fc).tolist()])

print(forecasts)