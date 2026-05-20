import pandas as pd
import numpy as np
import random
from xgboost import XGBRegressor

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'], index_col='DATE')

# Drop entirely empty columns as decided in Turn 1
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True, errors='ignore')

# Primary target
target = '% WEIGHTED ILI'
series = df[target].copy()

# Chronological 80/20 split: first 1050 rows train, rest test
train_series = series.iloc[:1050]
test_series = series.iloc[1050:]

H = len(test_series)  # forecast horizon = 263

# Create lag features for training (52 lags)
def create_lag_features(s, n_lags):
    X, y = [], []
    for i in range(n_lags, len(s)):
        X.append(s[i - n_lags:i].values)
        y.append(s[i])
    return np.array(X), np.array(y)

X_train, y_train = create_lag_features(train_series.values, n_lags=52)

# Train XGBoost model once with fixed hyperparameters
model = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42,
                     objective='reg:squarederror', verbosity=0)
model.fit(X_train, y_train)

# Recursive one-step-ahead forecasting over the entire test set
last_window = list(train_series.values[-52:])  # initial window of 52 most recent training values
forecasts = []

for step in range(H):
    X_input = np.array(last_window[-52:]).reshape(1, -1)
    pred = model.predict(X_input)[0]
    forecasts.append(pred)
    # Update window with prediction only; no ground truth is used
    last_window.append(pred)
    last_window = last_window[-52:]  # keep window size fixed at 52

print(forecasts)