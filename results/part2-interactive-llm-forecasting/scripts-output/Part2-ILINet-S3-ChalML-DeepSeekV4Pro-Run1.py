import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# Drop completely empty columns
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True)

# Parse time index, sort, set weekly frequency
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
df = df.set_index('DATE').asfreq('W-SAT')  # most dates are Saturdays; handle gaps with NaN

# Target column
target_col = '% WEIGHTED ILI'
series = df[target_col].copy()

# Split point: 1050 train, 263 test (0-based indices 0..1049 train, 1050..1312 test)
train_size = 1050
test_size = len(series) - train_size
test_indices = list(range(train_size, len(series)))

# Hyperparameters
lags = 13
n_estimators = 50
max_depth = 4
learning_rate = 0.1
H = 4

def create_lag_features(y, lags):
    # given a series y, returns a DataFrame of lag features from the last lags points of y
    X = pd.DataFrame()
    for lag in range(1, lags+1):
        X[f'lag_{lag}'] = y.shift(lag)
    X = X.iloc[lags:]  # drop rows with NaN lags
    return X

def train_model(X, y):
    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbose=-1
    )
    model.fit(X, y)
    return model

def recursive_forecast(model, last_lags, horizon):
    # last_lags: list/array of length lags, with last_lags[-1] being the most recent known value
    # returns list of length horizon with recursive predictions
    forecast_vals = []
    current_lags = list(last_lags)
    for _ in range(horizon):
        # Create input DataFrame with column names matching training
        input_df = pd.DataFrame([current_lags], columns=[f'lag_{i+1}' for i in range(lags)])
        pred = model.predict(input_df)[0]
        forecast_vals.append(pred)
        # shift lags: drop oldest, append prediction
        current_lags = current_lags[1:] + [pred]
    return forecast_vals

forecasts = []

# Rolling update: retrain at each test step, use ground truth after prediction
for idx in test_indices:
    # Use all data before current index for training
    train_series = series.iloc[:idx]
    # Build features from training part
    X_train = create_lag_features(train_series, lags)
    y_train = train_series.iloc[lags:]  # align with X_train indices
    # Train model
    model = train_model(X_train, y_train)
    # Get the last lags from the training series (most recent known values)
    last_lags = train_series.iloc[-lags:].tolist()
    # Recursive forecast H steps
    h_step_forecasts = recursive_forecast(model, last_lags, H)
    # We only need the first forecast (1-step ahead) for this test point
    forecast_val = h_step_forecasts[0]
    forecasts.append(forecast_val)
    # Ground truth is available for this index after prediction; it will be used in next iteration's training

# Ensure length matches test set
assert len(forecasts) == test_size

print(forecasts)