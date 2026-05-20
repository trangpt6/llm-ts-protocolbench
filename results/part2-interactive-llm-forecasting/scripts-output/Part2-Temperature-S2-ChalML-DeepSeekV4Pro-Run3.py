import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
y = df['Daily minimum temperatures'].values

# Train/test split (80% chronological)
n_total = len(y)
n_train = int(0.8 * n_total)  # 2921
train_end_idx = n_train - 1  # last train index (0-based)
test_start_idx = n_train     # first test index

# Forecasting setup: one-step ahead, rolling update with retraining after each step
# Lags: 7
forecasts = []

# Pre-allocate target series (use copy to avoid modifying original)
current_y = y[:n_train].tolist()  # will be expanded step by step

# First train on initial training set
X_train = []
y_train = []
for i in range(7, len(current_y)):
    X_train.append(current_y[i-7:i])
    y_train.append(current_y[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# Predict first test point
last_lags = current_y[-7:]  # last 7 values of training set
pred = model.predict(np.array([last_lags]))[0]
forecasts.append(pred)

# Now expand with true test value and repeat for remaining test points
for t in range(test_start_idx + 1, n_total):
    # Append true value of previous test point (ground truth for t-1)
    true_val = y[t-1]
    current_y.append(true_val)
    
    # Rebuild training data with expanded current_y (up to index t-1)
    X_train = []
    y_train = []
    for i in range(7, len(current_y)):
        X_train.append(current_y[i-7:i])
        y_train.append(current_y[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Retrain model
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Predict next test point
    last_lags = current_y[-7:]
    pred = model.predict(np.array([last_lags]))[0]
    forecasts.append(pred)

# Output the list
print(forecasts)