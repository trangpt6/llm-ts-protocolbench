import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
random.seed(42)
np.random.seed(42)

# Read the data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'])
df.sort_values('Month', inplace=True)
df.set_index('Month', inplace=True)
series = df['Passengers'].copy()

# Chronological split: first 115 train, last 29 test
train_len = 115
train = series.iloc[:train_len]
test = series.iloc[train_len:]

# Fixed hyperparameters
lags = 12
n_estimators = 50
max_depth = 4
H = 12  # forecast horizon

def create_lagged_features(s, lags):
    # Given a time series, produce X (lag features) and y (next value)
    n = len(s)
    X = np.zeros((n - lags, lags))
    y = np.zeros(n - lags)
    for i in range(lags, n):
        X[i - lags, :] = s.iloc[i - lags : i].values  # values from i-lags to i-1
        y[i - lags] = s.iloc[i]
    return X, y

def recursive_forecast(model, last_values, H):
    forecasts = []
    last_vals = last_values.copy()
    for _ in range(H):
        pred = model.predict(last_vals.reshape(1, -1))[0]
        forecasts.append(pred)
        # shift window: drop oldest, append prediction
        last_vals = np.append(last_vals[1:], pred)
    return forecasts

# Rolling forecasting
all_train_series = train.values.copy()  # will grow with true test values
forecasts = []
for i in range(len(test)):
    # Build training set from current all_train_series (all true past data)
    s = pd.Series(all_train_series)
    X_train, y_train = create_lagged_features(s, lags)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Last known values for forecasting (last lags values)
    last_vals = all_train_series[-lags:]
    
    # Multi-step forecast
    pred_H = recursive_forecast(model, last_vals, H)
    forecasts.append(pred_H[0])  # first step prediction
    
    # Update ground truth: append true test value
    all_train_series = np.append(all_train_series, test.iloc[i])

# Convert to list of floats
forecasts = [float(x) for x in forecasts]

# Output
print(forecasts)