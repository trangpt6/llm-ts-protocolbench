import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing: drop entirely empty columns
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True, errors='ignore')

# Primary target
target_col = '% WEIGHTED ILI'

# Extract target series
y = df[target_col].values

# Total timesteps
n_total = len(y)

# Train/test split
train_size = int(0.8 * n_total)  # 1050
test_size = n_total - train_size  # 263

# Fixed hyperparameters
lags = 13
n_estimators = 50
max_depth = 4
learning_rate = 0.1

# Prepare initial training data: we need lagged features up to index (train_size - 1)
# For recursive forecasting, we'll create lags from the known series portion

def create_lagged_features(series, lags):
    # series is 1D array of all available history
    # returns X (len - lags, lags) and y (len - lags,) for training
    X = []
    y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

# Forecasting loop
forecasts = []

# Current known history: up to end of training set
history = y[:train_size].copy()

for step in range(test_size):
    # Use all available history to train model
    train_y = history
    X_train, y_train = create_lagged_features(train_y, lags)
    
    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Prepare initial lags for forecast: last 'lags' values of known history
    current_lags = train_y[-lags:].copy()
    
    # Recursive 4-step ahead forecast (but we only need the first step)
    preds_4 = []
    for h in range(4):
        X_pred = current_lags.reshape(1, -1)
        next_pred = model.predict(X_pred)[0]
        preds_4.append(next_pred)
        # Update lags by shifting and appending prediction
        current_lags = np.append(current_lags[1:], next_pred)
    
    # The first step forecast corresponds to the current test index
    forecast_value = preds_4[0]
    forecasts.append(forecast_value)
    
    # Update history with the true value of this test point (ground truth enabled)
    true_val = y[train_size + step]
    history = np.append(history, true_val)

# Print the final forecast list
print(forecasts)