import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Split parameters
train_size = 115
test_size = len(y) - train_size

# Hyperparameters and setup
lags = 12
horizon = 12
block_size = 12

forecasts = []
current_train_end = train_size

# Block-wise rolling update with retraining
while current_train_end < len(y):
    y_train = y[:current_train_end]
    
    # Determine how many steps to predict in this block
    steps_to_predict = min(horizon, len(y) - current_train_end)
    
    block_preds = []
    
    # Direct multi-step strategy: train a separate model for each horizon step
    for h in range(1, steps_to_predict + 1):
        X_train = []
        Y_train = []
        
        # Create lagged features and target for step h
        for i in range(lags, len(y_train) - h + 1):
            X_train.append(y_train[i - lags : i])
            Y_train.append(y_train[i + h - 1])
            
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        # Initialize and train LightGBM model
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        model.fit(X_train, Y_train)
        
        # Predict step h
        X_test = np.array([y_train[-lags:]])
        pred = model.predict(X_test)[0]
        block_preds.append(pred)
        
    forecasts.extend(block_preds)
    
    # Advance the training window by the block size (use ground truth)
    current_train_end += steps_to_predict

print(forecasts)