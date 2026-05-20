import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Define train and test sizes
train_size = 115
test_size = len(y) - train_size

# Hyperparameters
lags = 12
n_estimators = 50
max_depth = 3
learning_rate = 0.1

# Function to create lag features
def create_features(data, lags):
    X, Y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        Y.append(data[i])
    return np.array(X), np.array(Y)

forecasts = []

# Rolling update with retraining
for i in range(test_size):
    # Current training data expands by 1 at each step
    current_train_data = y[:train_size + i]
    
    # Create features and target
    X_train, Y_train = create_features(current_train_data, lags)
    
    # Initialize and train the model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, Y_train)
    
    # Prepare the test instance
    X_test = current_train_data[-lags:].reshape(1, -1)
    
    # Predict one step ahead
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))

# Print the final forecast list
print(forecasts)