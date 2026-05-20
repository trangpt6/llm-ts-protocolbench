import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Train and test split
train_size = 115
test_size = len(y) - train_size
y_train_initial = y[:train_size]
y_test = y[train_size:]

# Hyperparameters
lags = 12
n_estimators = 200
max_depth = 5
learning_rate = 0.05
block_size = 12
horizon = 12

# Function to create dataset for direct multi-step forecasting
def create_dataset(ts, lags, horizon):
    X, Y = [], []
    for i in range(len(ts) - lags - horizon + 1):
        X.append(ts[i:(i + lags)])
        Y.append(ts[(i + lags):(i + lags + horizon)])
    return np.array(X), np.array(Y)

forecasts = []
current_train = list(y_train_initial)
test_idx = 0

# Rolling block-wise forecasting
while test_idx < test_size:
    steps_to_forecast = min(block_size, test_size - test_idx)
    
    X_train, Y_train = create_dataset(np.array(current_train), lags, horizon)
    
    base_model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, Y_train)
    
    X_test = np.array(current_train[-lags:]).reshape(1, -1)
    pred = model.predict(X_test)[0]
    
    forecasts.extend(pred[:steps_to_forecast].tolist())
    
    current_train.extend(y_test[test_idx:test_idx + steps_to_forecast])
    test_idx += steps_to_forecast

# Print final forecasts
print(forecasts)