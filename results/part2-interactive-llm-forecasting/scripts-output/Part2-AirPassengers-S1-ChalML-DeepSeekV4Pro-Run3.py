import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# Ensure chronological order
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Extract target series
series = df['Passengers'].values

# Train/test split: first 115 train, next 29 test
train_size = 115
train = series[:train_size]
test = series[train_size:]

# Create lagged features for training
def create_lagged_features(data, lags):
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i][::-1])  # order from t-1 to t-lags
        y.append(data[i])
    return np.array(X), np.array(y)

lags = 12
X_train, y_train = create_lagged_features(train, lags)

# Define LightGBM model with fixed hyperparameters (ignore multi_step_strategy N/A)
model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Recursive one-step ahead forecasting without ground truth updates
forecasts = []
# Start with the last `lags` values from the training set
current_window = list(train[-lags:])

for i in range(len(test)):
    # Predict next step using the current window (most recent lag at the end)
    X_pred = np.array(current_window[::-1]).reshape(1, -1)  # reverse to match t-1, t-2,... order
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    # Update window: drop oldest, append predicted value
    current_window.pop(0)
    current_window.append(pred)

print(forecasts)