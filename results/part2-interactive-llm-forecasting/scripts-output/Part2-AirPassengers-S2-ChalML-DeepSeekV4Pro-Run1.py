import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seed for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# No preprocessing needed

# Define target and time index
target = 'Passengers'
y = df[target].values

# Train/test split: first 80% (115) for training, rest (29) for test
train_size = 115
train = y[:train_size]
test = y[train_size:]  # length 29

# Lags for feature engineering
lags = 12

# Function to create lag features
def create_lag_features(series, lags):
    n = len(series)
    X = np.zeros((n - lags, lags))
    for i in range(lags, n):
        X[i - lags] = series[i - lags:i]
    return X

# Initial full train series (we'll extend it during rolling)
train_series = list(train.copy())

# Forecasts list
forecasts = []

# Rolling one-step ahead with retraining
for t in range(len(test)):
    # Prepare training data up to current point
    current_length = len(train_series)
    if current_length > lags:
        X_train = create_lag_features(np.array(train_series), lags)
        y_train = np.array(train_series[lags:])
    else:
        # In case insufficient data for lags (should not happen with train_size 115 > 12)
        X_train = np.empty((0, lags))
        y_train = np.array([])

    # Train XGBoost regressor
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    if len(X_train) > 0:
        model.fit(X_train, y_train)
    else:
        # Fallback naive forecast if no training data
        model = None

    # Prepare last lags for prediction
    if len(train_series) >= lags:
        last_lags = np.array(train_series[-lags:]).reshape(1, -1)
        if model is not None:
            pred = model.predict(last_lags)[0]
        else:
            pred = train_series[-1]  # naive
    else:
        pred = train_series[-1]  # fallback

    forecasts.append(pred)

    # Append true observed value (ground truth) to training series for next step
    train_series.append(test[t])

# Output the forecasts as flat list
print(forecasts)