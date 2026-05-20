import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds
random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
target_col = 'Daily minimum temperatures'
data = df[target_col].values

# Split data
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

lags = 14
block_size = 30
forecasts = []

history = list(train_data)

for i in range(0, len(test_data), block_size):
    # Create training dataset from history
    X_train = []
    y_train = []
    for j in range(lags, len(history)):
        X_train.append(history[j-lags:j])
        y_train.append(history[j])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    
    # Forecast block
    current_block_size = min(block_size, len(test_data) - i)
    current_history = history[-lags:].copy()
    
    for step in range(current_block_size):
        X_test = np.array(current_history[-lags:]).reshape(1, -1)
        pred = model.predict(X_test)[0]
        forecasts.append(float(pred))
        current_history.append(float(pred))
        
    # Update history with true values
    history.extend(test_data[i:i+current_block_size])

print(forecasts)