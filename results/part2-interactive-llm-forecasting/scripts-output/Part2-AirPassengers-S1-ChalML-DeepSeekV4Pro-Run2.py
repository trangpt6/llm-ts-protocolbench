import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Reproducibility seeds
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# Preprocessing decision: none needed
# Ensure Passengers is numeric
df['Passengers'] = pd.to_numeric(df['Passengers'])

# Chronological split
n = len(df)
train_size = int(0.8 * n)
train = df.iloc[:train_size].copy()
test_indices = df.iloc[train_size:].index  # indices of test rows

# Extract target values as numpy for easy manipulation
series = df['Passengers'].values.astype(float)

# Hyperparameters
lags = 12
n_estimators = 300
max_depth = 6
learning_rate = 0.05

# Create lagged features for training
def create_lagged_features(y, lookback):
    X, Y = [], []
    for i in range(lookback, len(y)):
        X.append(y[i-lookback:i])
        Y.append(y[i])
    return np.array(X), np.array(Y)

train_y = series[:train_size]
train_X, train_Y = create_lagged_features(train_y, lags)

# Train LightGBM regressor with fixed parameters
model = lgb.LGBMRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=42,
    verbose=-1
)
model.fit(train_X, train_Y)

# Recursive one-step ahead forecasting
# Start with the full history available up to end of training
history = list(series[:train_size])
predictions = []

# Generate forecasts for each test point
for i in range(len(test_indices)):
    # Last lags values from history
    last_lags = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(last_lags)[0]
    predictions.append(pred)
    # Append prediction to history for next step
    history.append(pred)

forecasts = predictions
print(forecasts)