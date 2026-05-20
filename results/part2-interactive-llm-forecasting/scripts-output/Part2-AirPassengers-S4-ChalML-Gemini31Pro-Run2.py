import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Setup parameters
train_size = 115
total_size = len(y)
lags = 12
block_size = 12
n_estimators = 200
max_depth = 5
learning_rate = 0.05

# Function to create features for direct multi-step forecasting
def create_direct_features(series, lags, step):
    X, Y = [], []
    for i in range(lags, len(series) - step + 1):
        X.append(series[i-lags:i])
        Y.append(series[i+step-1])
    return np.array(X), np.array(Y)

forecasts = []
current_train_end = train_size

# Block-wise rolling update with retraining
while current_train_end < total_size:
    steps_to_forecast = min(block_size, total_size - current_train_end)
    current_train_y = y[:current_train_end]
    
    block_forecasts = []
    # Direct strategy: train a separate model for each step in the horizon
    for h in range(1, steps_to_forecast + 1):
        X_train, Y_train = create_direct_features(current_train_y, lags, h)
        
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, Y_train)
        
        # Predict using the last 'lags' observations of the current training set
        X_test = current_train_y[-lags:].reshape(1, -1)
        pred = model.predict(X_test)[0]
        block_forecasts.append(pred)
        
    forecasts.extend(block_forecasts)
    current_train_end += steps_to_forecast

# Print final forecast list
print(forecasts)