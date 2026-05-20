import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

# Set random seeds for reproducibility
np.random.seed(42)

# Read CSV
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Preprocessing: replace artificial constant end-of-month days with NaN
daily_std = df['OT'].resample('D').transform('std')
df.loc[daily_std == 0, 'OT'] = np.nan

# Interpolate missing values linearly to obtain continuous series
df['OT'] = df['OT'].interpolate(method='linear')

# Split
train_size = 14016
train = df.iloc[:train_size]['OT']
test = df.iloc[train_size:]['OT']

# Feature engineering with lags=168
def create_lag_features(series, lags=168):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series.iloc[i-lags:i].values)
        y.append(series.iloc[i])
    return np.array(X), np.array(y)

# Initial training
X_train, y_train = create_lag_features(train)
model = lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# Forecasting
forecasts = []
known_series = list(train.values)  # keep as list for easy appending
horizon = 168
test_values = test.values
test_length = len(test_values)
block_start = 0

while block_start < test_length:
    remaining = test_length - block_start
    block_len = min(horizon, remaining)
    block_preds = []
    
    # Recursive multi-step forecasting
    recent_window = known_series[-168:]  # last 168 known values
    for _ in range(block_len):
        X_input = np.array(recent_window).reshape(1, -1)
        pred = model.predict(X_input)[0]
        block_preds.append(pred)
        # Update window: remove oldest, append prediction
        recent_window = recent_window[1:] + [pred]
    
    forecasts.extend(block_preds)
    
    # After block, update known_series with true test values for this block
    true_block = test_values[block_start:block_start+block_len]
    known_series.extend(true_block)
    
    # Retrain model using extended known_series
    extended_series = pd.Series(known_series)
    X_retrain, y_retrain = create_lag_features(extended_series)
    model = lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_retrain, y_retrain)
    
    block_start += block_len

# Ensure output length matches test set
forecasts = forecasts[:test_length]
print(forecasts)