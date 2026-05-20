import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# No preprocessing as per Turn 1

# Define train/test split sizes as per Turn 0
train_size = 158
test_size = 40

# Extract the target column (Ice cream) and exogenous column (Heater)
heater = df["Heater"].values
ice_cream = df["Ice cream"].values

# Create lagged features for training
lags = 12
X_train = []
y_train = []
for i in range(lags, train_size):
    heater_lags = heater[i-lags:i]
    ice_cream_lags = ice_cream[i-lags:i]
    X_train.append(np.concatenate([heater_lags, ice_cream_lags]))
    y_train.append(ice_cream[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train LightGBM model with specified hyperparameters
model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Prepare for recursive forecasting
# We need access to the full heater series (including test) because it is treated as exogenous known future
# Heater values for test set are available
full_heater = heater  # entire dataset
# We build an ice_cream history starting with training values, then append predictions
ice_cream_history = list(ice_cream[:train_size])

forecasts = []
for j in range(test_size):
    # Current index in the full dataset corresponds to train_size + j
    current_idx = train_size + j
    # Get the last 12 lags of heater (which include test heater values as we go)
    heater_lags = full_heater[current_idx - lags:current_idx]
    # Get the last 12 lags of ice cream from history (training + previous predictions)
    ice_cream_lags = ice_cream_history[current_idx - lags:current_idx]
    # Form feature vector
    X_input = np.concatenate([heater_lags, ice_cream_lags]).reshape(1, -1)
    # Predict the next ice cream value
    y_pred = model.predict(X_input)[0]
    forecasts.append(y_pred)
    # Append prediction to ice cream history for subsequent steps
    ice_cream_history.append(y_pred)

# Output the forecast list
print(forecasts)