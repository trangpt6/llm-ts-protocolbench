import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')
series = df['OT']

# identify and replace constant-value days (imputation artifacts) with NaN
daily_std = series.resample('D').std()
constant_dates = daily_std[daily_std == 0].index
for d in constant_dates:
    series.loc[d.strftime('%Y-%m-%d')] = np.nan

# split into train and test
total_len = len(series)
train_len = int(0.8 * total_len)
train_series = series.iloc[:train_len]
test_series = series.iloc[train_len:]

# parameters
lags = 168
horizon = 168

# function to create training matrix from a given series (ignoring NaN targets)
def build_train_data(ser):
    X, y = [], []
    vals = ser.values
    for i in range(lags, len(ser)):
        if np.isnan(vals[i]):
            continue
        X.append(vals[i-lags:i])
        y.append(vals[i])
    return np.array(X), np.array(y)

# initial training on train_series
X_train, y_train = build_train_data(train_series)

# lightgbm model with fixed hyperparameters
model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1
)
model.fit(X_train, y_train)

# prepare for recursive block-wise forecasting
test_values = test_series.values
history = train_series.values.copy()
forecasts = []

block_start = 0
while block_start < len(test_values):
    curr_horizon = min(horizon, len(test_values) - block_start)
    context = history[-lags:].copy().tolist()
    block_preds = []
    for _ in range(curr_horizon):
        feat = np.array(context[-lags:]).reshape(1, -1)
        pred = model.predict(feat)[0]
        block_preds.append(pred)
        context.append(pred)
    forecasts.extend(block_preds)
    # incorporate true block into history
    true_block = test_values[block_start:block_start+curr_horizon]
    history = np.concatenate([history, true_block])
    # retrain model with expanded history
    expanded_series = pd.Series(history)
    X_new, y_new = build_train_data(expanded_series)
    model.fit(X_new, y_new)
    block_start += curr_horizon

print(forecasts)