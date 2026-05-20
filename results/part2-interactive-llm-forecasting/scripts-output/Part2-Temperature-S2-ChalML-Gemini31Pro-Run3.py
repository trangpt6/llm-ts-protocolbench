import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing: Convert Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Extract target series
target_col = 'Daily minimum temperatures'
series = df[target_col].copy()

# Define split
total_timesteps = len(series)
train_size = int(0.8 * total_timesteps)
train_series = series.iloc[:train_size]
test_series = series.iloc[train_size:]

# Hyperparameters
lags = 7
model_params = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbose': -1
}

# Initialize history with training data
history = list(train_series.values)
test_values = test_series.values
forecasts = []

# Rolling update forecasting
for i in range(len(test_values)):
    # Create dataset from current history
    X_train = []
    y_train = []
    for j in range(lags, len(history)):
        X_train.append(history[j-lags:j])
        y_train.append(history[j])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train model
    model = lgb.LGBMRegressor(**model_params)
    model.fit(X_train, y_train)
    
    # Prepare input for prediction
    X_test = np.array(history[-lags:]).reshape(1, -1)
    
    # Predict one step ahead
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    
    # Update history with the true value
    history.append(test_values[i])

# Print final forecasts
print(forecasts)