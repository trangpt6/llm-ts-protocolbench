import pandas as pd
import numpy as np
import lightgbm as lgb

# Set random seeds for reproducibility
np.random.seed(42)
random_seed = 42

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Target and exogenous series
ice = df['Ice cream'].values
heater = df['Heater'].values

# Parameters
lags = 12
n_estimators = 300
max_depth = 6
learning_rate = 0.05

# Train/test split (chronological, no shuffling)
train_size = 158
y = ice

# Create lagged features for training
# For t from lags to train_size-1
X_train = []
y_train = []
for t in range(lags, train_size):
    # Ice cream lags: t-12 to t-1
    ice_lags = ice[t-lags:t][::-1]  # most recent first
    heater_lags = heater[t-lags:t][::-1]
    features = np.concatenate([ice_lags, heater_lags])
    X_train.append(features)
    y_train.append(ice[t])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=random_seed,
    verbose=-1
)
model.fit(X_train, y_train)

# Recursive forecasting for test period
test_size = len(ice) - train_size
forecasts = []
# Initial ice window: last 12 actual values from training
ice_window = ice[train_size - lags:train_size].tolist()  # length lags

for i in range(test_size):
    # Current heater window: next 12 actual heater values starting from
    # train_size + i - lags to train_size + i - 1
    heater_start = train_size + i - lags
    heater_end = train_size + i
    heater_window = heater[heater_start:heater_end][::-1]  # most recent first
    # Build feature vector
    ice_vals = ice_window[::-1]  # already most recent first from window
    features = np.concatenate([ice_vals, heater_window]).reshape(1, -1)
    # Predict next ice cream
    pred = model.predict(features)[0]
    forecasts.append(pred)
    # Update ice window: remove oldest, append prediction
    ice_window.pop(0)
    ice_window.append(pred)

print(forecasts)