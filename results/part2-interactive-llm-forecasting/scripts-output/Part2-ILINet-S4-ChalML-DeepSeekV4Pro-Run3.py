import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])

# Insert missing date 2002-01-06 with NaN values
new_row = pd.DataFrame({'DATE': pd.to_datetime(['2002-01-06']), '% WEIGHTED ILI': np.nan,
                       '%UNWEIGHTED ILI': np.nan, 'AGE 0-4': np.nan, 'AGE 25-49': np.nan,
                       'AGE 25-64': np.nan, 'AGE 5-24': np.nan, 'AGE 50-64': np.nan,
                       'AGE 65': np.nan, 'ILITOTAL': np.nan, 'NUM. OF PROVIDERS': np.nan,
                       'TOTAL PATIENTS': np.nan})
df = pd.concat([df, new_row], ignore_index=True)
df = df.sort_values('DATE').reset_index(drop=True)

# Set DATE as index
df.set_index('DATE', inplace=True)

# Replace summer zero blocks: target NaN where TOTAL PATIENTS is 0 or NaN
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].mask((df['TOTAL PATIENTS'] == 0) | (df['TOTAL PATIENTS'].isna()))

# Extract target series
y = df['% WEIGHTED ILI']

# Split by date according to Turn 0
y_train = y.loc[:'2017-11-19']
y_test = y.loc['2017-11-26':]

# Ensure y_test length is 263 as required
assert len(y_test) == 263

# Parameters from Turn 2
lags = 52
n_estimators = 300
max_depth = 5
learning_rate = 0.05
block_size = 52
horizon_length = 52

# Function to create training data for direct multi-step forecasting
def create_train_data(y_series, h):
    y_non_nan = y_series.dropna()
    X_list, y_list = [], []
    for i in range(lags, len(y_non_nan) - h + 1):
        X_list.append(y_non_nan.iloc[i-lags:i].values)
        y_list.append(y_non_nan.iloc[i + h - 1])
    return np.array(X_list), np.array(y_list)

# Function to train all 52 models
def train_models(y_series):
    models = {}
    for h in range(1, horizon_length + 1):
        X, y_target = create_train_data(y_series, h)
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators,
                                 max_depth=max_depth, learning_rate=learning_rate,
                                 random_state=42)
        model.fit(X, y_target)
        models[h] = model
    return models

# Initialize models on training data only (before any test data)
models = train_models(y_train)

# Forecasting loop
y_current = y_train.copy()
n_test = len(y_test)
forecasts = []
start_offset = 0

while start_offset < n_test:
    h_end = min(block_size, n_test - start_offset)
    last_52 = y_current.iloc[-52:].values.reshape(1, -1)
    block_pred = []
    for h in range(1, h_end + 1):
        pred = models[h].predict(last_52)[0]
        block_pred.append(pred)
    forecasts.extend(block_pred)
    start_offset += h_end
    if start_offset >= n_test:
        break
    true_block = y_test.iloc[start_offset - block_size : start_offset].values
    y_current = pd.concat([y_current, pd.Series(true_block, index=y_test.index[start_offset - block_size : start_offset])])
    models = train_models(y_current)

print(forecasts)