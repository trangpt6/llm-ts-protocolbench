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
target_col = '% WEIGHTED ILI'

# Preprocessing: Handle missing date 2002-01-06
df = df.set_index('DATE')
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_range)
df[target_col] = df[target_col].interpolate(method='linear')

# Preprocessing: Fill missing exogenous values with 0
df['AGE 25-49'] = df['AGE 25-49'].fillna(0)
df['AGE 50-64'] = df['AGE 50-64'].fillna(0)
# Fill other columns to ensure no NaNs interfere with training
df = df.fillna(0)

# Feature Engineering: Create 13 lags as specified in hyperparameters
lags = 13
for i in range(1, lags + 1):
    df[f'lag_{i}'] = df[target_col].shift(i)

# Drop rows with NaNs from lagging
df = df.dropna()

# Define train and test periods based on Turn 0
# Train period: 1997-10-12 to 2017-10-08
# Test period: 2017-10-15 to 2022-10-09
train_end = pd.to_datetime('2017-10-08')
test_start = pd.to_datetime('2017-10-15')
test_end = pd.to_datetime('2022-10-09')

# Fixed Hyperparameters
params = {
    'n_estimators': 50,
    'max_depth': 4,
    'learning_rate': 0.1,
    'random_state': 42,
    'importance_type': 'split',
    'verbosity': -1
}

feature_cols = [f'lag_{i}' for i in range(1, lags + 1)]
forecasts = []

# Rolling forecast with retraining at every step
test_dates = pd.date_range(start=test_start, end=test_end, freq='W-SUN')

for current_date in test_dates:
    # Training set: all data before the current test date
    train_df = df[df.index < current_date]
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    # Initialize and train model
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    # Predict one-step ahead
    # The features for the current_date prediction are the lags available at that timestamp
    X_test = df.loc[[current_date], feature_cols]
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))

print(forecasts)