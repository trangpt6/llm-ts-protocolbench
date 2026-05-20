import pandas as pd
import numpy as np
import xgboost as xgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing from Turn 1
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')
# Resample to strict weekly frequency and insert missing weeks
df = df.resample('W-SUN').asfreq()
# Replace exact 0.0 values in target with NaN
df.loc[df['% WEIGHTED ILI'] == 0.0, '% WEIGHTED ILI'] = np.nan
# Linear interpolation for missing values
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# Define target and split from Turn 0
target = '% WEIGHTED ILI'
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# Feature Engineering: Lags
lags = 52
def create_lags(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series.iloc[i-n_lags:i].values)
        y.append(series.iloc[i])
    return np.array(X), np.array(y)

X_train, y_train = create_lags(train_df[target], lags)

# Fixed Model and Hyperparameters from Turn 2
# multi_step_strategy: 'N/A' is ignored
params = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'random_state': 42,
    'objective': 'reg:squarederror'
}

model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# Recursive Forecasting Setup from Turn 2
# Static setup: train once, recursive inference, no ground truth used
forecasts = []
# Initial input comes from the last 'lags' values of the training set
current_input = train_df[target].iloc[-lags:].values.tolist()

for i in range(test_size):
    # Reshape input for XGBoost
    X_input = np.array(current_input[-lags:]).reshape(1, -1)
    # Predict one step ahead
    pred = model.predict(X_input)[0]
    forecasts.append(float(pred))
    # Append prediction to input for next recursive step
    current_input.append(pred)

# Output the final forecast list
print(forecasts)