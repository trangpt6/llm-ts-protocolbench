import random
import numpy as np
import pandas as pd
import xgboost as xgb

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Extract target series
y = df['Daily minimum temperatures'].astype(float).values

# Chronological split
train_size = 2921
y_train = y[:train_size]
y_test = y[train_size:]

lags = 14
block_size = 30
test_len = len(y_test)

# Determine blocks
num_full_blocks = test_len // block_size
remainder = test_len % block_size
blocks = [block_size] * num_full_blocks
if remainder > 0:
    blocks.append(remainder)

def make_lag_features(series, lags):
    n = len(series)
    X = np.zeros((n - lags, lags))
    for i in range(lags):
        X[:, i] = series[i:n - lags + i]
    y_out = series[lags:]
    return X, y_out

forecasts = []
history = y_train.copy()

for b, h in enumerate(blocks):
    # Build features and train on current history
    X_tr, y_tr = make_lag_features(history, lags)
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_tr, y_tr)
    
    # Recursive multi-step forecast for this block
    window = history[-lags:].copy()
    block_preds = []
    for step in range(h):
        X_pred = window.reshape(1, -1)
        pred = model.predict(X_pred)[0]
        block_preds.append(float(pred))
        window = np.append(window[1:], pred)
    
    forecasts.extend(block_preds)
    
    # Append true values of this block to history for retraining
    start = sum(blocks[:b])
    end = start + h
    history = np.concatenate([history, y_test[start:end]])

print(forecasts)