import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

# Set random seeds
random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.sort_index(inplace=True)

# Features
data = df[['Heater', 'Ice cream']].values

# Setup parameters
train_size = 158
test_size = 40
lags = 6
horizon = 12

# Initialize forecasts list
forecasts = []

# Rolling update and retraining
for i in range(test_size):
    # Current train data
    current_train = data[:train_size + i]
    
    # Create training samples
    X_train = []
    y_train = []
    for t in range(lags, len(current_train)):
        X_train.append(current_train[t-lags:t].flatten())
        y_train.append(current_train[t])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Recursive forecasting
    current_window = current_train[-lags:].copy()
    
    step_forecasts = []
    for h in range(horizon):
        X_test = current_window.flatten().reshape(1, -1)
        pred = model.predict(X_test)[0]
        
        # Append Ice cream prediction
        step_forecasts.append(pred[1])
        
        # Update window
        current_window = np.vstack([current_window[1:], pred])
        
    forecasts.extend(step_forecasts)

print(forecasts)