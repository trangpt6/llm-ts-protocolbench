import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target = '% WEIGHTED ILI'
y = df[target].values

# Split: first 1044 train, last 261 test
train_size = 1044
test_size = 261
train_y = y[:train_size]
test_y = y[train_size:train_size+test_size]

# Create lag features for a given array (1D)
def create_lags(arr, lags=13):
    X, y_target = [], []
    for i in range(lags, len(arr)):
        X.append(arr[i-lags:i])
        y_target.append(arr[i])
    return np.array(X), np.array(y_target)

# Initialize storage for predictions
forecasts = []

# Rolling retrain with ground truth update
current_train = train_y.copy()
for i in range(test_size):
    # Build training set from all data up to current point
    # For i=0, current_train = train_y only
    X_train, y_train = create_lags(current_train, lags=13)
    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        objective='regression',
        verbose=-1
    )
    model.fit(X_train, y_train)
    # Predict next step: use last 13 values of current_train
    last_lags = current_train[-13:].reshape(1, -1)
    pred = model.predict(last_lags)[0]
    forecasts.append(pred)
    # Update current_train with true value from test set
    current_train = np.append(current_train, test_y[i])

print(forecasts)