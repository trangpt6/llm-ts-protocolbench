import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
df = df.rename(columns={'Daily minimum temperatures': 'temp'})
full_dates = pd.date_range(start='1981-01-01', end='1990-12-31', freq='D')
df = df.set_index('Date').reindex(full_dates).rename_axis('Date').reset_index()
df['temp'] = df['temp'].ffill()
y_all = df['temp'].values
n_total = len(y_all)

# Split train/test based on Turn 0 train end date
train_cutoff_date = pd.Timestamp('1988-12-30')
train_idx = df.index[df['Date'] <= train_cutoff_date].tolist()
test_idx = df.index[df['Date'] > train_cutoff_date].tolist()

last_train = train_idx[-1]
n_train = len(train_idx)

# Prepare lag features (lags 1 to 7) for all indices
lags = 7
X_full = np.full((n_total, lags), np.nan)
for lag in range(1, lags + 1):
    X_full[lag:, lag - 1] = y_all[:-lag]

forecasts = []

# Loop over test points
for i in test_idx:
    # current test point index i corresponds to date we need to forecast
    # training includes all up to i-1 (including last known actual)
    cur_train_end = i - 1

    # Build training samples for each horizon h=1..7
    # We'll train separate LGBM models for each horizon
    models = {}
    for h in range(1, lags + 1):
        # target index = t + h
        mask = (np.arange(lags, cur_train_end - h + 1) >= lags)  # ensure we have features for index t where t >= lags
        t_indices = np.arange(lags, cur_train_end - h + 1)
        if len(t_indices) == 0:
            raise ValueError("Not enough training data for horizon {}".format(h))
        X_train = X_full[t_indices, :]
        y_train = y_all[t_indices + h]
        model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                                  random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        models[h] = model

    # Forecast for horizon 1 using the most recent available feature row
    # The feature row is from index i-1 (the last known day)
    X_pred = X_full[i - 1:i, :]  # shape (1, lags)
    y_pred = models[1].predict(X_pred)[0]
    forecasts.append(y_pred)

# Print forecasts as flat list on the last line
print(forecasts)