import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# Read and prepare data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()
target_col = 'Ice cream'
exog_col = 'Heater'

# Chronological split: 80% train, 20% test (rows=198)
train_end = '2017-02'
test_start = '2017-03'

# Feature engineering function
def create_lag_features(data, target_col, exog_col, lag=6):
    df_feat = data.copy()
    for col in [target_col, exog_col]:
        for i in range(1, lag+1):
            df_feat[f'{col}_lag{i}'] = df_feat[col].shift(i)
    df_feat = df_feat.dropna()
    X = df_feat[[f'{target_col}_lag{i}' for i in range(1, lag+1)] +
                [f'{exog_col}_lag{i}' for i in range(1, lag+1)]]
    y = df_feat[target_col]
    return X, y

# Recursive multi-step forecast function
def recursive_forecast(model, last_lags, horizon, exog_series_future):
    # last_lags is a dictionary with keys: target_lag1..target_lag6, exog_lag1..exog_lag6
    # exog_series_future: list/array of future Heater values for next horizon steps
    preds = []
    current_lags = last_lags.copy()
    for h in range(horizon):
        # Form feature vector
        feat = []
        for i in range(1, 6+1):
            feat.append(current_lags[f'{target_col}_lag{i}'])
        for i in range(1, 6+1):
            feat.append(current_lags[f'{exog_col}_lag{i}'])
        feat = np.array(feat).reshape(1, -1)
        # Predict next target
        next_val = model.predict(feat)[0]
        preds.append(next_val)
        # Update lags for next step
        # shift target lags: lag1 becomes next_val, old lag1 becomes lag2, etc.
        # shift exog lags: lag1 gets known future exog value, rest shift
        new_target_lags = {}
        new_exog_lags = {}
        new_target_lags[f'{target_col}_lag1'] = next_val
        for i in range(2, 6+1):
            new_target_lags[f'{target_col}_lag{i}'] = current_lags[f'{target_col}_lag{i-1}']
        new_exog_lags[f'{exog_col}_lag1'] = exog_series_future[h]
        for i in range(2, 6+1):
            new_exog_lags[f'{exog_col}_lag{i}'] = current_lags[f'{exog_col}_lag{i-1}']
        current_lags = {**new_target_lags, **new_exog_lags}
    return preds

# Initial training data
train_data = df.loc[:train_end]
X_train, y_train = create_lag_features(train_data, target_col, exog_col, lag=6)

# Model hyperparameters
rf_params = {'n_estimators': 50, 'max_depth': 4}

# Rolling forecast loop over test set months
test_months = df.loc[test_start:].index
forecasts = []
current_train_end = train_end
# We need to forecast each month sequentially
for forecast_month in test_months:
    # Train model on current training set
    X, y = create_lag_features(df.loc[:current_train_end], target_col, exog_col, lag=6)
    model = RandomForestRegressor(random_state=42, **rf_params)
    model.fit(X, y)
    # Prepare the last known lags (from the end of training set)
    # The last row in training set corresponds to time current_train_end
    # We need lags that start from the previous 6 observations.
    # The features for predicting the next step (forecast_month) are built from the last 6 actual values.
    # So we take the last 6 observations of both series up to current_train_end.
    last_obs = df.loc[:current_train_end].iloc[-6:]
    last_lags = {}
    # The most recent observation is at current_train_end; lag1 is that observation.
    # We'll fill lag i with the value i steps before current_train_end.
    for i in range(1, 6+1):
        idx = -i  # from end: -1 = most recent
        last_lags[f'{target_col}_lag{i}'] = last_obs[target_col].iloc[idx]
        last_lags[f'{exog_col}_lag{i}'] = last_obs[exog_col].iloc[idx]
    # Known future Heater values for the next 12 months (since horizon=12)
    # Get Heater values from forecast_month onward for 12 steps
    future_heater = df.loc[forecast_month:].head(12)[exog_col].values.tolist()
    # Compute recursive 12-step forecast
    preds_12 = recursive_forecast(model, last_lags, horizon=12, exog_series_future=future_heater)
    # We only need the 1-step ahead forecast (first element)
    forecasts.append(preds_12[0])
    # Update current_train_end to include the actual value of forecast_month
    current_train_end = forecast_month
    # (The actual target for forecast_month is now known and automatically included in df for next retraining)

print(forecasts)