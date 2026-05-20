import pandas as pd
import numpy as np
import lightgbm as lgb
import random
import warnings

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
warnings.filterwarnings('ignore')

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Split data
train_size = int(0.8 * len(y))
train_data = y[:train_size]
test_size = len(y) - train_size

# Hyperparameters
lags = 12
n_estimators = 300
max_depth = 6
learning_rate = 0.05

# Create training features and target
X_train = []
y_train = []
for i in range(lags, train_size):
    X_train.append(train_data[i-lags:i])
    y_train.append(train_data[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Initialize and train model
model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
model.fit(X_train, y_train)

# Recursive forecasting
forecasts = []
current_history = list(train_data[-lags:])

for _ in range(test_size):
    X_test = np.array(current_history[-lags:]).reshape(1, -1)
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    current_history.append(pred)

# Print final forecasts
print(forecasts)