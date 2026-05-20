import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# Preprocessing from Turn 1
# 1. Handle missing timestamp 2002-01-06 by resampling and interpolating
df = df.set_index('DATE')
df = df.resample('W-SUN').asfreq()
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# 2. Fill missing exogenous columns with 0
df['AGE 25-49'] = df['AGE 25-49'].fillna(0)
df['AGE 50-64'] = df['AGE 50-64'].fillna(0)
# Fill other potential NaNs in exogenous columns to ensure model runs
df = df.fillna(0)

target = '% WEIGHTED ILI'
lags = 52
n_estimators = 300
max_depth = 5
learning_rate = 0.05
block_size = 52
test_size = 261
train_size = 1044

# Feature engineering: Create lag features
def create_lags(data, n_lags):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Forecasting process
all_data = df[target].values
forecasts = []

# Block-wise rolling update logic
# We iterate through the test set in blocks of 52
for start_idx in range(train_size, train_size + test_size, block_size):
    end_idx = min(start_idx + block_size, train_size + test_size)
    current_horizon = end_idx - start_idx
    
    # Training data: all data available before the current block
    train_data_current = all_data[:start_idx]
    
    # Direct Multi-step Strategy: Train a separate model for each step in the horizon
    block_forecasts = np.zeros(current_horizon)
    
    for h in range(1, current_horizon + 1):
        # For lead time h, target is y[t+h], features are y[t], y[t-1], ..., y[t-51]
        X_train, y_train = [], []
        # We can only train up to the point where we have a target at lead time h
        for i in range(lags, len(train_data_current) - h + 1):
            X_train.append(train_data_current[i-lags:i])
            y_train.append(train_data_current[i + h - 1])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=1
        )
        model.fit(X_train, y_train)
        
        # Predict for the specific step in the test block
        # The features are the last 52 observations before the block starts
        X_test = train_data_current[-lags:].reshape(1, -1)
        block_forecasts[h-1] = model.predict(X_test)[0]
        
    forecasts.extend(block_forecasts.tolist())

# Ensure the output matches the test size exactly
print(forecasts[:test_size])