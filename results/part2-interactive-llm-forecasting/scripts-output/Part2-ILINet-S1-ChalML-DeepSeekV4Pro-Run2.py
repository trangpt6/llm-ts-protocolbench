import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/ILINet.csv')

# Drop the completely empty column AGE 25-49
if 'AGE 25-49' in df.columns:
    df.drop(columns=['AGE 25-49'], inplace=True)

# Ensure DATE is datetime and sorted
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

# Extract target series
y = df['% WEIGHTED ILI'].values

# Train/test split: first 1050 train, rest test
train_size = 1050
y_train = y[:train_size]
y_test = y[train_size:]

# Create lag features for training
lags = 52
def create_lag_features(series, lags):
    X = []
    Y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i][::-1])  # Most recent first
        Y.append(series[i])
    return np.array(X), np.array(Y)

X_train, Y_train = create_lag_features(y_train, lags)

# Train XGBoost model with fixed hyperparameters
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    objective='reg:squarederror'
)
model.fit(X_train, Y_train)

# Recursive one-step ahead forecasting over test set
# Initial input is last lags values from training set
last_known = y_train[-lags:].tolist()
forecasts = []

for _ in range(len(y_test)):
    # Form input from most recent lags
    x_input = np.array(last_known[-lags:][::-1]).reshape(1, -1)
    pred = model.predict(x_input)[0]
    forecasts.append(pred)
    # Append prediction to history for next step
    last_known.append(pred)

# Convert forecasts to list of Python floats
forecasts_list = [float(f) for f in forecasts]

# Output the forecast list
print(forecasts_list)