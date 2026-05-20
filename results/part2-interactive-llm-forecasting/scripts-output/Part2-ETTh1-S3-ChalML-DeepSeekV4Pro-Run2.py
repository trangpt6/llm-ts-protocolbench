import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Preprocessing: negative values -> NaN
df.loc[df['OT'] < 0, 'OT'] = np.nan

# Preprocessing: constant-value days (all 24 hours identical) -> NaN
daily_std = df['OT'].resample('D').std()
constant_days = daily_std[daily_std == 0].index
for day in constant_days:
    df.loc[day.strftime('%Y-%m-%d'), 'OT'] = np.nan

# Interpolate linearly
df['OT'] = df['OT'].interpolate(method='linear')

# Chronological split
train_size = int(0.8 * len(df))
train = df.iloc[:train_size]['OT'].copy()
test = df.iloc[train_size:]['OT'].copy()

# Function to create lag features for a given series (up to given index) for direct multi-step
def create_lag_features(series, lags=12, horizon=1):
    # series is a Series up to current time
    X, y = [], []
    # Ensure we have enough history
    for i in range(lags - 1, len(series) - horizon):
        X.append(series.iloc[i - lags + 1 : i + 1].values)
        y.append(series.iloc[i + horizon])
    return np.array(X), np.array(y)

# Rolling forecasting
forecasts = []
# We'll iterate over each test time step
test_indices = range(len(test))
# For convenience, maintain full series so far (train + observed test)
full_series = train.tolist()  # start with training

for i in test_indices:
    # Current available history (train + previously observed test points)
    # Note: full_series currently contains up to time index train_size-1 + i (since previous test point already appended)
    current_series = pd.Series(full_series)
    
    # Prepare models for all 24 horizons
    horizon_models = []
    for h in range(1, 25):
        X_tr, y_tr = create_lag_features(current_series, lags=12, horizon=h)
        if len(X_tr) == 0:
            # fallback: not enough data to train, skip (should not happen with this size)
            continue
        model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, objective='reg:squarederror', verbosity=0)
        model.fit(X_tr, y_tr)
        horizon_models.append(model)
    
    # Predict the next 24 values
    # Features: last 12 values of current_series
    if len(current_series) < 12:
        # should not happen
        forecasts.append(np.nan)
    else:
        last_lags = current_series.iloc[-12:].values.reshape(1, -1)
        # Only need horizon 1 forecast for the evaluation
        pred_h1 = horizon_models[0].predict(last_lags)[0]
        forecasts.append(pred_h1)
    
    # Update with true observation (ground truth enabled)
    # The true value for this test point is test.iloc[i]
    true_val = test.iloc[i]
    full_series.append(true_val)

print(forecasts)