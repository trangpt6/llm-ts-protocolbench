import pandas as pd
import numpy as np
import xgboost as xgb
import random
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')

# Preprocessing: detect constant last-day-of-month blocks
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
last_day_mask = df.index == df.index.shift(1, freq='D').asfreq('H', method='pad') + pd.DateOffset(days=1) - pd.DateOffset(hours=1)
# But simpler: group by date and check if day is last day of month
def is_last_day_of_month(day, month, year):
    try:
        next_day = pd.Timestamp(year=year, month=month, day=day) + pd.DateOffset(days=1)
        return next_day.month != month
    except:
        return False

mask_last_day = df.index.to_series().apply(lambda x: is_last_day_of_month(x.day, x.month, x.year))
# For each date that is last day of month, check if OT values are constant across that day
df['date_only'] = df.index.date
for date_val, group in df.groupby('date_only'):
    if mask_last_day.loc[group.index[0]]:
        if group['OT'].std() == 0:
            df.loc[group.index, 'OT'] = np.nan
# Interpolate linearly
df['OT'] = df['OT'].interpolate(method='linear')
# Drop helper columns
df.drop(['day','month','year','date_only'], axis=1, inplace=True)

# Split train/test
train_size = 14016
train = df.iloc[:train_size]['OT']
test = df.iloc[train_size:]['OT']

# Function to create lag features for a given series
def create_lag_features(series, lags=12):
    if len(series) <= lags:
        return pd.DataFrame(columns=[f'lag_{i}' for i in range(1, lags+1)])
    data = pd.DataFrame()
    for i in range(1, lags+1):
        data[f'lag_{i}'] = series.shift(i)
    data = data.iloc[lags:]
    return data

# Rolling forecast
all_series = train.copy()
forecasts = []

lags = 12
Horizon = 24

for i in range(len(test)):
    # Current observed series length
    N = len(all_series)
    # Prepare training data for each horizon model
    preds_h = []
    for h in range(1, Horizon+1):
        # Build training data: use all data up to N-1 (excluding the unknown future)
        # For horizon h, target is series[t+h] for t from lags-1 to N-1-h
        # So we need to align
        train_features = []
        train_target = []
        for t in range(lags, N - h + 1):
            # features are series[t-lags:t]
            features = all_series[t-lags:t].values
            train_features.append(features)
            train_target.append(all_series[t+h-1])
        if len(train_features) == 0:
            # fallback if not enough data
            last_values = all_series[-lags:].values
            preds_h.append(last_values[-1])  # naive persistence
            continue
        X = np.array(train_features)
        y = np.array(train_target)
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbosity=0)
        model.fit(X, y)
        # Predict next step: use the last lags values
        last_features = all_series[-lags:].values.reshape(1, -1)
        pred_h = model.predict(last_features)[0]
        preds_h.append(pred_h)
    # Store the 1-step-ahead forecast
    forecasts.append(preds_h[0])
    # Append true test value to series
    all_series = pd.concat([all_series, pd.Series([test.iloc[i]])], ignore_index=True)

print(forecasts)