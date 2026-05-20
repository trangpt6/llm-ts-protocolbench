import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Define parameters
lags = 12
train_size = 115
test_size = len(y) - train_size

# Prepare forecasts list
forecasts = []

# Rolling forecast with retraining
for i in range(test_size):
    # Current training data
    current_y = y[:train_size + i]
    
    # Create lag features for training
    X_train = []
    Y_train = []
    for j in range(lags, len(current_y)):
        X_train.append(current_y[j-lags:j])
        Y_train.append(current_y[j])
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    # Initialize and train model
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, Y_train)
    
    # Prepare test instance
    X_test = np.array([current_y[-lags:]])
    
    # Predict
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))

# Print final forecasts
print(forecasts)