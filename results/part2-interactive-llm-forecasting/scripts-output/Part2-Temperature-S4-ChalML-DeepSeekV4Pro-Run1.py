import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False, index_col='Date')
target_col = 'Daily minimum temperatures'
df[target_col] = pd.to_numeric(df[target_col])

n_total = len(df)
n_train = 2921
train_series = df[target_col].iloc[:n_train].values
test_series = df[target_col].iloc[n_train:].values

n_lags = 14
block_size = 30
horizon = 30
n_estimators = 200
max_depth = 5
learning_rate = 0.05

def create_lagged_dataset(series, n_lags):
    X_list = []
    y_list = []
    for i in range(n_lags, len(series)):
        X_list.append(series[i-n_lags:i])
        y_list.append(series[i])
    return np.array(X_list), np.array(y_list)

def recursive_forecast(model, last_lags, steps):
    preds = []
    current_lags = list(last_lags)
    for _ in range(steps):
        X_pred = np.array(current_lags).reshape(1, -1)
        next_val = model.predict(X_pred)[0]
        preds.append(next_val)
        current_lags.pop(0)
        current_lags.append(next_val)
    return preds

X_train, y_train = create_lagged_dataset(train_series, n_lags)
model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, random_state=42)
model.fit(X_train, y_train)

known_series = list(train_series)
forecasts = []

num_blocks = len(test_series) // block_size
remainder = len(test_series) % block_size

for b in range(num_blocks):
    last_lags = known_series[-n_lags:]
    block_forecasts = recursive_forecast(model, last_lags, block_size)
    forecasts.extend(block_forecasts)
    start_idx = b * block_size
    end_idx = start_idx + block_size
    true_block = test_series[start_idx:end_idx]
    known_series.extend(true_block)
    X_retrain, y_retrain = create_lagged_dataset(np.array(known_series), n_lags)
    model.fit(X_retrain, y_retrain)

if remainder > 0:
    last_lags = known_series[-n_lags:]
    block_forecasts = recursive_forecast(model, last_lags, remainder)
    forecasts.extend(block_forecasts)

print(forecasts)