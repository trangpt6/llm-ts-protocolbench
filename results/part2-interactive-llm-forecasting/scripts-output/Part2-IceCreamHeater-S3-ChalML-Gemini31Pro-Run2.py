import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# Extract the two features
data = df[['Heater', 'Ice cream']].values

# Define split
train_size = int(0.8 * len(data))
test_size = len(data) - train_size

# Hyperparameters
lags = 6
n_estimators = 50
max_depth = 4
horizon = 12

forecasts = []

# Rolling update over the test set
for i in range(test_size):
    # History available up to the current step
    history = data[:train_size + i]
    
    # Prepare training data
    X_train = []
    y_train = []
    for j in range(lags, len(history)):
        X_train.append(history[j-lags:j].flatten())
        y_train.append(history[j])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Recursive forecasting for the horizon
    current_lags = history[-lags:].copy()
    step_forecasts = []
    for h in range(horizon):
        X_test = current_lags.flatten().reshape(1, -1)
        pred = model.predict(X_test)[0]
        # Append the prediction for the primary target
        step_forecasts.append(pred[1])
        
        # Update lags with the predicted values for both features
        current_lags = np.vstack((current_lags[1:], pred))
        
    # Store the forecasts
    forecasts.extend(step_forecasts)

# Print the final flat list of forecasts
print(forecasts)