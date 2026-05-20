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
y = df['% WEIGHTED ILI'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
y_train = y[:train_size]
y_test = y[train_size:]
test_size = len(y_test)

block_size = 52
horizon = 52

forecasts = []
history = list(y_train)
i = 0
while i < test_size:
    steps = min(horizon, test_size - i)
    model = SARIMAX(
        history,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 52),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=steps)
    forecasts.extend([float(v) for v in np.asarray(fc).flatten()])
    history.extend(list(y_test[i:i + steps]))
    i += steps

forecasts = [float(x) for x in forecasts]
print(forecasts)