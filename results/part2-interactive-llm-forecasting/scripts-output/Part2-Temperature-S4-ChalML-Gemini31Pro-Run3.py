import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
y = df['Daily minimum temperatures'].values

# Train/test split
train_size = int(0.8 * len(y))
train_y = y[:train_size]
test_y = y[train_size:]

# Hyperparameters and setup
lags = 14
n_estimators = 200
max_depth = 5
learning_rate = 0.05
block_size = 30

# Feature creation function
def create_features(data, lags):
    X, Y = [], []
    for i in range(len(data) - lags):
        X.append(data[i:i+lags])
        Y.append(data[i+lags])
    return np.array(X), np.array(Y)

# Forecasting loop
history = list(train_y)
forecasts = []

for i in range(0, len(test_y), block_size):
    current_horizon = min(block_size, len(test_y) - i)
    
    # Create training data from current history
    X_train, Y_train = create_features(history, lags)
    
    # Initialize and train model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, Y_train)
    
    # Recursive forecasting for the current block
    current_history = history[-lags:].copy()
    block_forecasts = []
    
    for step in range(current_horizon):
        x_input = np.array(current_history[-lags:]).reshape(1, -1)
        pred = model.predict(x_input)[0]
        block_forecasts.append(float(pred))
        current_history.append(float(pred))
        
    forecasts.extend(block_forecasts)
    
    # Update history with true values for the next block's training
    history.extend(test_y[i:i+current_horizon])

# Print final forecasts
print(forecasts)