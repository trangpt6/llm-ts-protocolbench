import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(float)

# Setup parameters
train_size = 115
test_size = len(data) - train_size
lags = 12
horizon = 12

forecasts = []

# Rolling update and retraining
for i in range(test_size):
    current_t = train_size + i
    train_data = data[:current_t]
    
    # Feature engineering lag features
    X_train = []
    y_train = []
    for t in range(lags, len(train_data)):
        X_train.append(train_data[t-lags:t])
        y_train.append(train_data[t])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Recursive multi-step forecasting
    current_window = list(train_data[-lags:])
    step_preds = []
    for _ in range(horizon):
        pred = model.predict([current_window])[0]
        step_preds.append(pred)
        current_window = current_window[1:] + [pred]
        
    # Collect the 1-step ahead prediction to cover the test set
    forecasts.append(step_preds[0])

# Print final forecast list
print(forecasts)