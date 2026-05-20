import random
import numpy as np
import pandas as pd
import xgboost as xgb

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset from local file
df = pd.read_csv(r'../../../data/Temperature.csv')

# preprocessing: parse date and ensure numeric target
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce')

# chronological train/test split
train_size = 2921
train_series = df['Daily minimum temperatures'].iloc[:train_size].reset_index(drop=True)
test_series = df['Daily minimum temperatures'].iloc[train_size:].reset_index(drop=True)

lags = 14
block_size = 30
n_test = len(test_series)
forecasts = []

def create_lag_features(series, lags):
    values = series.values
    n = len(values)
    X = []
    y = []
    for i in range(lags, n):
        X.append(values[i - lags:i])
        y.append(values[i])
    return np.array(X), np.array(y)

history = train_series.copy()
while len(forecasts) < n_test:
    current_horizon = min(block_size, n_test - len(forecasts))
    X_train, y_train = create_lag_features(history, lags)
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)
    last_lags = history.iloc[-lags:].values.copy()
    block_preds = []
    for _ in range(current_horizon):
        X_pred = last_lags.reshape(1, -1)
        pred = model.predict(X_pred)[0]
        block_preds.append(float(pred))
        last_lags = np.append(last_lags[1:], pred)
    forecasts.extend(block_preds)
    start_idx = len(forecasts) - current_horizon
    end_idx = len(forecasts)
    true_block = test_series.iloc[start_idx:end_idx]
    history = pd.concat([history, true_block], ignore_index=True)

print(forecasts)