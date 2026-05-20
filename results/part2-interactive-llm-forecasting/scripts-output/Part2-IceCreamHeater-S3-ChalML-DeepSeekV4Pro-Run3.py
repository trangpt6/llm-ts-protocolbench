import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
icecream = df['Ice cream'].values
heater = df['Heater'].values
n = len(icecream)

# Chronological split: train first 158, test next 40
train_size = 158
test_size = 40

# Fixed hyperparameters
lags = 6
n_estimators = 50
max_depth = 4

# Helper function to create features from a given series history up to a given index
# series_hist: array of values up to index t-1
def create_features(ice_hist, heat_hist, t):
    # t is the index for which we need features (the target time step)
    # we assume history contains values up to t-1
    features = []
    for lag in range(1, lags+1):
        idx = t - lag
        features.append(ice_hist[idx])
        features.append(heat_hist[idx])
    return np.array(features).reshape(1, -1)

# Build initial training data using indices 0..train_size-1 as history
# Train target indices from lags to train_size-1
X_train = []
y_train = []
for i in range(lags, train_size):
    feat = create_features(icecream[:i], heater[:i], i)
    X_train.append(feat)
    y_train.append(icecream[i])
X_train = np.vstack(X_train)
y_train = np.array(y_train)

# Initialize and train model
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# Rolling forecast
forecasts = []
# full history arrays, start with training data
ice_hist = icecream[:train_size].tolist()
heat_hist = heater[:train_size].tolist()

for i in range(train_size, n):
    # Predict next value using current history (index i is the target time step)
    X_pred = create_features(np.array(ice_hist), np.array(heat_hist), i)
    y_pred = model.predict(X_pred)[0]
    forecasts.append(y_pred)

    # Append true value to history (ground truth available at inference)
    ice_hist.append(icecream[i])
    heat_hist.append(heater[i])

    # Retrain model using extended data up to current time index i
    X_new = []
    y_new = []
    # Now history length is i+1 (indices 0..i)
    # target indices from lags to i
    for j in range(lags, i+1):
        feat = create_features(np.array(ice_hist), np.array(heat_hist), j)
        X_new.append(feat)
        y_new.append(ice_hist[j])
    X_new = np.vstack(X_new)
    y_new = np.array(y_new)

    model.fit(X_new, y_new)

# Output the forecast list
print(forecasts)