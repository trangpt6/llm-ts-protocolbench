import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'

total = len(df)
train_size = int(0.8 * total)
original_test_dates = pd.DatetimeIndex(df['DATE'].iloc[train_size:].values)
train_end_date = pd.Timestamp(df['DATE'].iloc[train_size - 1])

df_idx = df.set_index('DATE')[[target_col]]
full_idx = pd.date_range(start=df_idx.index.min(), end=df_idx.index.max(), freq='W-SUN')
df_full = df_idx.reindex(full_idx)
df_full[target_col] = df_full[target_col].interpolate(method='linear')

y = df_full[target_col].values.astype(float)
dates = df_full.index

lags = 52
X_list = []
y_list = []
d_list = []
for i in range(lags, len(y)):
    X_list.append(y[i - lags:i])
    y_list.append(y[i])
    d_list.append(dates[i])
X_arr = np.array(X_list)
y_arr = np.array(y_list)
d_arr = pd.DatetimeIndex(d_list)

train_mask = d_arr <= train_end_date
X_train = X_arr[train_mask]
y_train = y_arr[train_mask]

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=1,
    verbosity=0
)
model.fit(X_train, y_train)

train_end_pos = int(np.where(dates == train_end_date)[0][0])
window = list(y[train_end_pos - lags + 1: train_end_pos + 1])

test_dates_full = dates[dates > train_end_date]
preds = {}
for d in test_dates_full:
    x_in = np.array(window[-lags:]).reshape(1, -1)
    p = float(model.predict(x_in)[0])
    preds[pd.Timestamp(d)] = p
    window.append(p)

forecasts = [preds[pd.Timestamp(d)] for d in original_test_dates]

print(forecasts)