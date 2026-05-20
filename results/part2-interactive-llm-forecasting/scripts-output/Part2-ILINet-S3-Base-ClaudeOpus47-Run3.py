import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target = df['% WEIGHTED ILI'].astype(float).copy()
dates = df['DATE']

mask = (dates >= '1998-05-31') & (dates <= '2002-10-06') & (target == 0.0)
target.loc[mask] = np.nan
target = target.interpolate(method='linear').bfill().ffill()

n = len(target)
train_size = int(0.8 * n)
y_train = target.iloc[:train_size].values.astype(float)
y_test = target.iloc[train_size:].values.astype(float)
N = len(y_test)
H = 4

forecasts = []
history = list(y_train)

for i in range(N):
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False, maxiter=50)
    fc = res.forecast(steps=H)
    forecasts.extend([float(x) for x in np.asarray(fc)])
    history.append(float(y_test[i]))

print(forecasts)