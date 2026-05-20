import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# reproducibility
random.seed(42)
np.random.seed(42)

df_raw = pd.read_csv(r'../../../data/ILINet.csv')
df_raw['DATE'] = pd.to_datetime(df_raw['DATE'])
df_raw = df_raw.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'
tp_col = 'TOTAL PATIENTS'

train_size = 1040
test_size = 261
test_dates = list(df_raw['DATE'].iloc[train_size:train_size + test_size])
train_end_date = df_raw['DATE'].iloc[train_size - 1]
test_start_date = df_raw['DATE'].iloc[train_size]

full_index = pd.date_range(start=df_raw['DATE'].iloc[0], end=df_raw['DATE'].iloc[-1], freq='7D')
df_full = df_raw.set_index('DATE').reindex(full_index)

mask = (df_full[tp_col].fillna(0) == 0) & (df_full[target_col].fillna(-1) == 0)
df_full.loc[mask, target_col] = np.nan
y_full = df_full[target_col].interpolate(method='linear').ffill().bfill()

y_train_series = y_full.loc[:train_end_date]
y_test_series = y_full.loc[test_start_date:]

current_train = y_train_series.values.astype(float).copy()
y_test_vals = y_test_series.values.astype(float)

block_size = 52
forecasts_full = []
n_test = len(y_test_vals)
i = 0
while i < n_test:
    steps = min(block_size, n_test - i)
    model = SARIMAX(current_train,
                    order=(2, 0, 2),
                    seasonal_order=(1, 1, 1, 52),
                    trend='n',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.forecast(steps=steps)
    forecasts_full.extend([float(v) for v in np.asarray(fc)])
    current_train = np.concatenate([current_train, y_test_vals[i:i + steps]])
    i += steps

fc_series = pd.Series(forecasts_full, index=y_test_series.index)
forecasts = [float(fc_series.loc[d]) for d in test_dates]

print(forecasts)