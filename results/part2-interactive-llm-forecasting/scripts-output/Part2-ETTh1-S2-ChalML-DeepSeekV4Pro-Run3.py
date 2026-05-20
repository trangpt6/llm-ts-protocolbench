import random
import numpy as np
import pandas as pd
import xgboost as xgb

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# List of dates with constant-value artifacts (all hours of the day)
bad_dates = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31',
    '2018-03-31', '2018-05-31'
]
for d in bad_dates:
    if d in df.index.strftime('%Y-%m-%d'):
        df.loc[d, 'OT'] = np.nan

# Chronological split
train_size = 14016
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Function to create lag features from a Series
def create_lags(series, n_lags=12):
    df = pd.DataFrame({'y': series})
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    return df

# Build initial training set (only non-NaN rows)
train_lagged = create_lags(train['OT'], n_lags=12)
train_clean = train_lagged.dropna()
X_train_init = train_clean[[f'lag_{i}' for i in range(1,13)]].values
y_train_init = train_clean['y'].values

# Initial model
model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train_init, y_train_init)

# Rolling forecasting
# We'll maintain a full series combining train and observed test points.
full_series = pd.concat([train['OT'], test['OT']])
# Start of test period in the full series index
test_start_idx = len(train)  # = 14016

# Accumulator for valid test rows (features + target) for retraining
test_features_list = []
test_target_list = []

forecasts = []

for t in range(test_start_idx, len(full_series)):
    # Prepare features from the 12 previous hours (values before current t)
    # Since we haven't observed the true value at t yet (and will not use it)
    # The features are the values at indices t-1 down to t-12.
    feat_indices = [t - i for i in range(1, 13)]
    features = full_series.iloc[feat_indices].values.reshape(1, -1)  # may contain NaN
    # Predict
    pred = model.predict(features)[0]
    forecasts.append(pred)

    # Now we have the true value at t (test set ground truth)
    true_val = test['OT'].iloc[t - test_start_idx]

    # Update the full series with the true value (could be NaN)
    full_series.iloc[t] = true_val

    # If true value is valid (not NaN), add to retraining set
    if not np.isnan(true_val):
        # Need to capture the same features used for prediction, but they used old full_series values before updating t?
        # Actually the features used before the update are based on the series up to t-1, which is correct.
        # We'll store the features and the true target exactly as they were observed.
        test_features_list.append(features.flatten())
        test_target_list.append(true_val)

    # Retrain model on all available non-NaN data (init train + test valid so far)
    if len(test_target_list) > 0:
        X_retrain = np.vstack([X_train_init, np.array(test_features_list)])
        y_retrain = np.concatenate([y_train_init, np.array(test_target_list)])
    else:
        X_retrain = X_train_init
        y_retrain = y_train_init
    # Retrain from scratch each step (no incremental)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_retrain, y_retrain)

print(forecasts)