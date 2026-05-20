import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# Apply preprocessing steps
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
y = df['Passengers'].values

# Define split and parameters
train_size = 115
test_size = len(y) - train_size
lags = 12
horizon = 12

forecasts = []

# Rolling update with retraining at every time step
for i in range(test_size):
    y_train = y[:train_size + i]
    
    X_train = []
    Y_train = []
    for j in range(lags, len(y_train)):
        X_train.append(y_train[j-lags:j])
        Y_train.append(y_train[j])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, Y_train)
    
    # Recursive multi-step forecasting
    current_history = list(y_train[-lags:])
    step_forecasts = []
    for h in range(horizon):
        x_input = np.array(current_history[-lags:]).reshape(1, -1)
        pred = float(model.predict(x_input)[0])
        step_forecasts.append(pred)
        current_history.append(pred)
        
    forecasts.extend(step_forecasts)

# Print final forecast list
print(forecasts)