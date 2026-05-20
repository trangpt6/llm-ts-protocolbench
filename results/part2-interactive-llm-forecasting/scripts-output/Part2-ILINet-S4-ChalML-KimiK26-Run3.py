import pandas as pd
import numpy as np
import xgboost as xgb
import random

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)

target = '% WEIGHTED ILI'

# no preprocessing needed per Turn 1

total = len(df)
train_size = int(0.8 * total)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

lags = 52
n_estimators = 300
max_depth = 5
learning_rate = 0.05

def create_lagged_data(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series.iloc[i-lags:i].values)
        y.append(series.iloc[i])
    return np.array(X), np.array(y)

# initial training on full training set
X_train, y_train = create_lagged_data(train_df[target], lags)
model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
model.fit(X_train, y_train)

# rolling block-wise forecasting
block_size = 52
horizon = 52
forecasts = []
current_train_series = train_df[target].values.copy()  # keep as numpy array for efficiency
current_test_series = test_df[target].values.copy()

idx = 0  # start index in test set
while idx < len(test_df):
    # determine actual horizon for this block (last block may be shorter)
    actual_horizon = min(horizon, len(test_df) - idx)
    
    # create lagged features from the last `lags` values of the combined training+observed test history
    combined = np.concatenate([current_train_series, current_test_series[:idx]])
    if len(combined) < lags:
        break
    # build initial features for first step
    last_lags = combined[-lags:].reshape(1, -1)
    block_forecasts = []
    # multi-step direct forecast: predict each step sequentially using last lags
    for step in range(actual_horizon):
        pred = model.predict(last_lags)[0]
        block_forecasts.append(pred)
        # update lag window with predicted value (for next step within block)
        last_lags = np.roll(last_lags, -1)
        last_lags[0, -1] = pred
    
    forecasts.extend(block_forecasts)
    idx += actual_horizon
    
    # after each block, retrain model with all available true data
    # update current_train_series with true values from this block
    true_block = current_test_series[idx - actual_horizon:idx]
    current_train_series = np.concatenate([current_train_series, true_block])
    
    # retrain on all data including newly observed true values
    # need to recreate lagged features from the extended training series
    if len(current_train_series) > lags:
        X_retrain, y_retrain = [], []
        for i in range(lags, len(current_train_series)):
            X_retrain.append(current_train_series[i-lags:i])
            y_retrain.append(current_train_series[i])
        X_retrain = np.array(X_retrain)
        y_retrain = np.array(y_retrain)
        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        model.fit(X_retrain, y_retrain)

# Ensure forecasts length matches test size (261)
forecasts = forecasts[:len(test_df)]
print(forecasts)