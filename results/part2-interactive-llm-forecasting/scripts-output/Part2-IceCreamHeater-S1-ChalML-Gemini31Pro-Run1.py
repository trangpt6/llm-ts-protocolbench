import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# Preprocessing: Convert Month to datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Split data into train and test
train_size = 158
train_data = df.iloc[:train_size].copy()
test_data = df.iloc[train_size:].copy()

# Hyperparameters
lags = 12
n_estimators = 300
max_depth = 6
learning_rate = 0.05

# Prepare training data using lags of both Heater and Ice cream
X_train = []
y_train = []

target = train_data['Ice cream'].values
exog = train_data['Heater'].values

for i in range(lags, len(train_data)):
    feat = []
    for j in range(1, lags + 1):
        feat.append(target[i - j])
        feat.append(exog[i - j])
    X_train.append(feat)
    y_train.append(target[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train the LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Recursive forecasting over the test set
forecasts = []
current_target_history = list(target)
current_exog_history = list(exog) + list(test_data['Heater'].values)

for i in range(len(test_data)):
    feat = []
    for j in range(1, lags + 1):
        feat.append(current_target_history[-j])
        idx = len(train_data) + i - j
        feat.append(current_exog_history[idx])
    
    pred = model.predict(np.array([feat]))[0]
    forecasts.append(float(pred))
    current_target_history.append(pred)

# Print the final forecast list
print(forecasts)