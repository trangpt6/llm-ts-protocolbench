import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing: Parse Date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define target
target_col = 'Daily minimum temperatures'
y = df[target_col].values

# Train/test split
train_size = 2921
train_data = y[:train_size]
test_data = y[train_size:]

# Create lag features for training
lags = 14
X_train = []
y_train = []
for i in range(lags, len(train_data)):
    X_train.append(train_data[i-lags:i])
    y_train.append(train_data[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Initialize and train the XGBoost model
model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Recursive forecasting over the test set
forecasts = []
current_lags = list(train_data[-lags:])

for i in range(len(test_data)):
    X_test = np.array([current_lags])
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    current_lags.append(float(pred))
    current_lags.pop(0)

# Print the final forecast list
print(forecasts)