import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

# Set random seeds
np.random.seed(42)
random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Split parameters
train_size = 115
test_size = 29
lags = 12
horizon = 12

# Store all forecasts
forecasts = []

# Rolling update forecasting
for i in range(test_size):
# Define current training window
    train_data = y[:train_size + i]
    
# Create lag features
    X_train = []
    y_train = []
    for j in range(lags, len(train_data)):
        X_train.append(train_data[j-lags:j])
        y_train.append(train_data[j])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
# Train model
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
# Recursive multi-step forecast
    current_lags = train_data[-lags:].tolist()
    step_preds = []
    for h in range(horizon):
        pred = model.predict([current_lags])[0]
        step_preds.append(pred)
        current_lags.append(pred)
        current_lags.pop(0)
        
# Append to flat list
    forecasts.extend(step_preds)

# Print final flat list
print(forecasts)