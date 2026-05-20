import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

# Preprocessing: Reindex to complete weekly frequency
all_weeks = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(all_weeks)

# Preprocessing: Linear interpolation for primary target and constant fill for exogenous
target_col = '% WEIGHTED ILI'
df[target_col] = df[target_col].interpolate(method='linear')
df = df.fillna(0)

# Feature Engineering: Create 13 lags for the primary target
lags = 13
for i in range(1, lags + 1):
    df[f'lag_{i}'] = df[target_col].shift(i)

# Drop rows with NaN lags created by shifting
df = df.dropna()

# Define train and test sizes based on Turn 0 (80/20 split of original 1301 rows)
# Total timesteps after reindexing might change slightly, but we follow the Turn 0 split point.
# Original total_timesteps = 1301. Train size = 1040. Test size = 261.
# Since we dropped 13 rows for lags, we adjust the indices.
train_size = 1040
test_size = 261

# Prepare features and target
features = [f'lag_{i}' for i in range(1, lags + 1)]
X = df[features].values
y = df[target_col].values

# Forecasting loop: One-step ahead, rolling update, retrain at every step
forecasts = []
params = {
    'n_estimators': 50,
    'max_depth': 4,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbosity': -1
}

# The test set starts at index 1040 (relative to the original start)
# We iterate through the test period
for i in range(test_size):
    # Current training data: all data up to the start of the current test point
    curr_train_idx = train_size + i
    X_train = X[:curr_train_idx]
    y_train = y[:curr_train_idx]
    
    # Current test point features
    X_test = X[curr_train_idx : curr_train_idx + 1]
    
    # Initialize and train model
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Predict one step ahead
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))

# Output the final forecast list
print(forecasts)