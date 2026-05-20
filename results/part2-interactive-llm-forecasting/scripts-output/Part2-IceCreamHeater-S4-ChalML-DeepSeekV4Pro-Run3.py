import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()

target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']

train_size = 158
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

LAGS = 12
N_ESTIMATORS = 200
MAX_DEPTH = 5
LEARNING_RATE = 0.05
BLOCK_SIZE = 12
HORIZON = 12

def create_direct_features(series_df, lag=12, horizon=12):
    data = series_df.copy()
    for col in feature_cols:
        for i in range(0, lag):
            data[f'{col}_lag{i}'] = data[col].shift(i)
    data = data.dropna()
    results = {}
    for h in range(1, horizon+1):
        y = data[target_col].shift(-h)
        valid_idx = y.notna()
        feat_cols = [f'{col}_lag{i}' for col in feature_cols for i in range(0, lag)]
        X_h = data.loc[valid_idx, feat_cols]
        y_h = y[valid_idx]
        results[h] = (X_h, y_h)
    return results

true_vals_so_far = df.iloc[:train_size].copy()
forecasts = []
test_index = 0
num_test = len(test_df)

while test_index < num_test:
    remaining = num_test - test_index
    block_len = min(BLOCK_SIZE, remaining)
    # Train models on current data
    feature_data = create_direct_features(true_vals_so_far, lag=LAGS, horizon=HORIZON)
    models = {}
    for h in range(1, HORIZON+1):
        X_h, y_h = feature_data[h]
        model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=42,
            verbose=-1
        )
        model.fit(X_h, y_h)
        models[h] = model
    # Build forecast feature vector from last LAGS rows
    last_known = true_vals_so_far.iloc[-LAGS:]
    feature_vec = {}
    for col in feature_cols:
        for i in range(0, LAGS):
            feature_vec[f'{col}_lag{i}'] = last_known[col].iloc[-(i+1)]
    feature_row = pd.DataFrame([feature_vec])
    # Forecast for required horizon
    block_forecasts = []
    for h in range(1, block_len+1):
        pred = models[h].predict(feature_row)[0]
        block_forecasts.append(pred)
    forecasts.extend(block_forecasts)
    # Append true block and advance
    true_block = test_df.iloc[test_index:test_index+block_len]
    true_vals_so_far = pd.concat([true_vals_so_far, true_block])
    test_index += block_len

print(forecasts)