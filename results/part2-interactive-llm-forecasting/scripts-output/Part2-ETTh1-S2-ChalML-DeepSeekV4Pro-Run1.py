import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the raw CSV
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Preprocessing: detect constant-value days (all 24 hours same)
# Group by date (day) and check if all values are identical
df['day'] = df.index.date
constant_days = df.groupby('day')['OT'].transform('nunique') == 1
# Replace those constant values with NaN
df.loc[constant_days, 'OT'] = np.nan
# Linear interpolation to fill
df['OT'] = df['OT'].interpolate(method='linear')
df.drop(columns=['day'], inplace=True)

# Train/test split
total = len(df)
train_size = int(0.8 * total)
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Feature engineering: create lag features of order 12
def create_lag_features(series, lags=12):
    data = pd.DataFrame({'y': series})
    for lag in range(1, lags+1):
        data[f'lag_{lag}'] = data['y'].shift(lag)
    data.dropna(inplace=True)
    return data

# Prepare training data initially
train_series = train['OT']
train_data = create_lag_features(train_series)
X_train = train_data.drop(columns=['y'])
y_train = train_data['y']

# XGBoost model parameters (multi_step_strategy N/A ignored)
model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)

# Fit initial model
model.fit(X_train, y_train)

# Rolling forecast process
forecasts = []
# We need to maintain the series of all known values: concatenated train and observed test values
known_series = list(train_series)
# The last 12 known values before test start are the last 12 of train
# We'll predict for each point in test set

for i in range(len(test)):
    # Use the last 12 known values to form the feature row for next prediction
    if len(known_series) < 12:
        raise ValueError("Not enough lags for first prediction")
    last_12 = known_series[-12:]
    features = pd.DataFrame([last_12[::-1]], columns=[f'lag_{j}' for j in range(1,13)])  # lag_1 oldest? The order: lag_1 is t-1, lag_2 is t-2, etc. We'll align with training: original create_lag_features used shift 1 for lag_1 (t-1), shift 2 for lag_2 (t-2). So row: t-1,t-2,...,t-12. So we need features with lag_1 = last known (t-1), lag_2 = t-2, ..., lag_12 = t-12. So features = {lag_1: known_series[-1], lag_2: known_series[-2], ..., lag_12: known_series[-12]}. So we'll construct accordingly.
    feature_values = {f'lag_{j}': known_series[-j] for j in range(1,13)}
    features = pd.DataFrame([feature_values])
    # Predict
    pred = model.predict(features)[0]
    forecasts.append(pred)
    
    # Observe true test value
    true_val = test.iloc[i]['OT']
    known_series.append(true_val)
    
    # Retrain the model on all available data (train + observed test points up to now)
    # We'll recreate the full dataset of known series and rebuild features
    current_series = pd.Series(known_series)
    current_data = create_lag_features(current_series)
    if len(current_data) > 0:
        X_retrain = current_data.drop(columns=['y'])
        y_retrain = current_data['y']
        model.fit(X_retrain, y_retrain)

print(forecasts)