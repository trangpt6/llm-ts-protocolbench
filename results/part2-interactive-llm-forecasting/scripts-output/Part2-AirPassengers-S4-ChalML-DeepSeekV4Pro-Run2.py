import pandas as pd
import numpy as np
import random
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
series = df['Passengers'].astype(float)

# Chronological split
train_size = 115
train_series = series.iloc[:train_size]
test_series = series.iloc[train_size:]

# Function to create direct multi-step training data
def create_direct_samples(ser, horizon=12, n_lags=12):
    X, y = [], []
    values = ser.values
    L = len(values)
    for t in range(n_lags, L - horizon + 1):
        X.append(values[t - n_lags : t])
        y.append(values[t : t + horizon])
    return np.array(X), np.array(y)

# Forecasting loop
horizon = 12
block_size = 12
history_train = train_series.copy()
predictions = []

while len(predictions) < len(test_series):
    # Prepare training data from current history
    X_train, y_train = create_direct_samples(history_train, horizon=horizon, n_lags=12)
    # Train direct multi-output LightGBM with fixed hyperparameters
    base_lgb = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    )
    model = MultiOutputRegressor(base_lgb)
    model.fit(X_train, y_train)
    
    # Build input for next step: last 12 values from current history
    last_vals = history_train.values[-12:].reshape(1, -1)
    pred_block = model.predict(last_vals).flatten()
    
    # Determine how many steps to keep from this block
    steps_needed = len(test_series) - len(predictions)
    keep = min(steps_needed, len(pred_block))
    predictions.extend(pred_block[:keep].tolist())
    
    # Use ground truth for this block to extend history (if any true test values exist)
    if keep > 0:
        true_block = test_series.iloc[len(predictions) - keep : len(predictions)]
        history_train = pd.concat([history_train, true_block])

# Final forecast as flat list
print(predictions)