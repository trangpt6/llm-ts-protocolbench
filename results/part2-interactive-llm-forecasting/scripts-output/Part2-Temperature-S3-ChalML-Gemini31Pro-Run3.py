import pandas as pd
import numpy as np
import lightgbm as lgb
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds
random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Extract target values
target_col = 'Daily minimum temperatures'
data = df[target_col].values

# Train/test split
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
lags = 7
horizon = 7

# Function to create features and target for direct multi-step strategy
def build_xy(series, lags, h):
    X, y = [], []
    for i in range(len(series) - lags - h + 1):
        X.append(series[i : i + lags])
        y.append(series[i + lags + h - 1])
    return np.array(X), np.array(y)

history = list(train_data)
forecasts = []

# Rolling update forecasting
for i in range(len(test_data)):
    current_history = np.array(history)
    
    step_forecasts = []
    for h in range(1, horizon + 1):
        X_train, y_train = build_xy(current_history, lags, h)
        
        model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        X_test = current_history[-lags:].reshape(1, -1)
        pred = model.predict(X_test)[0]
        step_forecasts.append(float(pred))
        
    forecasts.extend(step_forecasts)
    
    # Update history with the true value
    history.append(test_data[i])

print(forecasts)