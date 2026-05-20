import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

full_idx = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='7D')
df = df.set_index('DATE').reindex(full_idx)
df.index.name = 'DATE'

target = df['% WEIGHTED ILI'].copy()

mask_offseason = pd.Series(False, index=target.index)
offseason_ranges = [
    ('1998-05-31', '1998-10-04'),
    ('1999-05-30', '1999-10-03'),
    ('2000-05-28', '2000-10-01'),
    ('2001-05-27', '2001-09-30'),
    ('2002-06-02', '2002-10-06'),
]
for s, e in offseason_ranges:
    mask_offseason.loc[s:e] = True

target[mask_offseason] = np.nan
target = target.interpolate(method='linear', limit_direction='both')

n_total = len(target)
n_train = int(0.8 * n_total)
train = target.iloc[:n_train]
test = target.iloc[n_train:]

model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False,
)
res = model.fit(disp=False)

pred = res.get_forecast(steps=len(test))
forecasts = pred.predicted_mean.tolist()

print(forecasts)