import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

y = df['Daily minimum temperatures'].astype(float).values

n = len(y)
train_size = 2920
test_size = n - train_size

block_size = 30
horizon = 30

history = list(y[:train_size])
test = y[train_size:]

forecasts = []
i = 0
while i < test_size:
    steps = min(horizon, test_size - i)
    model = SARIMAX(
        history,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 365),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=steps)
    forecasts.extend([float(v) for v in np.asarray(fc).tolist()])
    history.extend(list(test[i:i + steps]))
    i += steps

print(forecasts)