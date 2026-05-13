import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset from local CSV file
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Define chronological train/test split
total_timesteps = len(y)
train_size = int(0.8 * total_timesteps)

# Lag feature engineering configuration
lags = 12

def make_lag_features(series, lags):
    n = len(series)
    X = []
    Y = []
    for i in range(lags, n):
        X.append(series[i - lags:i])
        Y.append(series[i])
    return np.array(X), np.array(Y)

# Initialize training history with the train set only
history = y[:train_size].tolist()
X_train, y_train = make_lag_features(np.array(history), lags)

# Initialize XGBoost with fixed hyperparameters
model = XGBRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# Rolling one-step ahead forecast over the test set
forecasts = []
for i in range(train_size, total_timesteps):
    x_input = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(x_input)[0]
    forecasts.append(float(pred))
    true_val = y[i]
    history.append(true_val)
    X_new, y_new = make_lag_features(np.array(history), lags)
    model.fit(X_new, y_new)

print(forecasts)