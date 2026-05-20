import random
import numpy as np
import pandas as pd
import xgboost as xgb

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing: Parse Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Target column
target_col = 'Daily minimum temperatures'
series = df[target_col].values

# Train/test split
train_size = 2921
test_size = len(series) - train_size
train_series = series[:train_size]
test_series = series[train_size:]

# Hyperparameters
lags = 14
n_estimators = 200
max_depth = 5
learning_rate = 0.05
block_size = 30

# Feature engineering function
def create_features(data, lags):
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

forecasts = []
history = list(train_series)

# Block-wise rolling update
for i in range(0, test_size, block_size):
    steps_to_forecast = min(block_size, test_size - i)
    
    # Create training data from current history
    X_train, y_train = create_features(history, lags)
    
    # Initialize and train model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    
    # Recursive forecasting for the block
    current_lags = history[-lags:]
    block_forecasts = []
    for _ in range(steps_to_forecast):
        X_pred = np.array(current_lags).reshape(1, -1)
        pred = model.predict(X_pred)[0]
        block_forecasts.append(float(pred))
        
        # Update lags with the prediction
        current_lags.append(float(pred))
        current_lags.pop(0)
        
    forecasts.extend(block_forecasts)
    
    # Update history with true values for the next block
    history.extend(test_series[i:i+steps_to_forecast])

# Print final forecast list
print(forecasts)