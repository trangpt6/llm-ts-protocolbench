import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# reproducibility
np.random.seed(42)
import random
random.seed(42)

# load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])

# Turn 1 preprocessing: replace constant end-of-month days with NaN
constant_days = [
    '2016-07-31','2016-08-31','2016-10-31','2016-12-31',
    '2017-01-31','2017-03-31','2017-05-31','2017-07-31',
    '2017-08-31','2017-10-31','2017-12-31','2018-01-31',
    '2018-03-31','2018-05-31'
]
df['date_only'] = df['date'].dt.date
df.loc[df['date_only'].astype(str).isin(constant_days), 'OT'] = np.nan
df.drop(columns=['date_only'], inplace=True)

# split
train_size = 14016
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# lag feature function
def create_lags(series, lags=24):
    df_lag = pd.DataFrame()
    for lag in range(1, lags+1):
        df_lag[f'lag_{lag}'] = series.shift(lag)
    return df_lag

# build training features and target
lag_cols = [f'lag_{i}' for i in range(1, 25)]
train_lags = create_lags(train['OT'])
train_feat = pd.concat([train[['OT']], train_lags], axis=1).dropna()

X_train = train_feat[lag_cols].values
y_train = train_feat['OT'].values

# model training
model = LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# recursive forecasting
# initial lag window: last 24 valid values from training set (original, not NaN)
train_ot = train['OT'].values
# find last 24 non-NaN values
valid_train = train['OT'].dropna().values
if len(valid_train) >= 24:
    recent = list(valid_train[-24:])
else:
    raise ValueError('Not enough valid training data for initial lags')

forecasts = []
for _ in range(len(test)):
    X_pred = np.array(recent).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    recent.append(pred)
    recent = recent[1:]  # keep length 24

print(forecasts)