import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Setup parameters
lags = 12
train_size = 115
total_size = len(y)

forecasts = []

# Rolling update with retraining at every step
for t in range(train_size, total_size):
    # Prepare training data up to time t using ground truth
    y_history = y[:t]
    
    X_train = []
    Y_train = []
    for i in range(lags, len(y_history)):
        X_train.append(y_history[i-lags:i])
        Y_train.append(y_history[i])
        
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    # Initialize and train model
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, Y_train)
    
    # Prepare test instance for one-step ahead prediction
    X_test = np.array([y_history[-lags:]])
    
    # Predict and store
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))

# Print final forecasts
print(forecasts)