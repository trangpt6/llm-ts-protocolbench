import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'

n_total_original = len(df)
train_size = int(0.8 * n_total_original)
original_test_dates = df['DATE'].iloc[train_size:].tolist()

full_idx = pd.date_range(start=df['DATE'].iloc[0], end=df['DATE'].iloc[-1], freq='7D')
df_full = df.set_index('DATE').reindex(full_idx)
df_full.index.name = 'DATE'
df_full[target_col] = df_full[target_col].interpolate(method='linear')

y = df_full[target_col].values.astype(float)
dates = pd.DatetimeIndex(df_full.index)

LAGS = 13
N_ESTIMATORS = 50
MAX_DEPTH = 4
LR = 0.1
HORIZON = 4

def make_features(series, lags):
    n = len(series)
    X = np.zeros((n - lags, lags))
    y_arr = np.zeros(n - lags)
    for i in range(lags, n):
        X[i - lags] = series[i - lags:i]
        y_arr[i - lags] = series[i]
    return X, y_arr

forecasts = []
for test_date in original_test_dates:
    pos = dates.get_loc(test_date)
    train_y = y[:pos]
    X_train, y_train = make_features(train_y, LAGS)
    model = LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LR,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    last_window = train_y[-LAGS:].tolist()
    preds = []
    for h in range(HORIZON):
        x_in = np.array(last_window[-LAGS:]).reshape(1, -1)
        p = float(model.predict(x_in)[0])
        preds.append(p)
        last_window.append(p)
    forecasts.append(preds[0])

print(forecasts)