import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# Preprocessing
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
y = df['Passengers'].values

# Train/test split
train_size = 115
train_y = y[:train_size]
test_y = y[train_size:]

# Hyperparameters
lags = 12
n_estimators = 300
max_depth = 6
learning_rate = 0.05

# Create training features and target
X_train = []
Y_train = []
for i in range(lags, len(train_y)):
    X_train.append(train_y[i-lags:i])
    Y_train.append(train_y[i])

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Initialize and train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=42
)
model.fit(X_train, Y_train)

# Recursive forecasting
forecasts = []
current_history = list(train_y[-lags:])

for _ in range(len(test_y)):
    X_test = np.array(current_history[-lags:]).reshape(1, -1)
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    current_history.append(pred)

# Print final forecast list
print(forecasts)