import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], index_col='Date')
target_col = 'Daily minimum temperatures'
series = df[target_col].astype(float)

# Chronological split
total_len = len(series)
train_size = int(0.8 * total_len)  # 2921
train_series = series.iloc[:train_size]
test_series = series.iloc[train_size:]

# Function to create lag features, returns X, y from a series
def create_lag_features(s, lags=14):
    df_lag = pd.DataFrame({f'lag_{i}': s.shift(i) for i in range(1, lags+1)})
    df_lag['y'] = s
    df_lag = df_lag.dropna()
    X = df_lag.drop(columns='y')
    y = df_lag['y']
    return X, y

# Initial training
lags = 14
X_train, y_train = create_lag_features(train_series, lags)

# Fixed hyperparameters
model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    objective='reg:squarederror',
    verbosity=0
)
model.fit(X_train, y_train)

# Forecast block-wise rolling
block_size = 30
horizon = 30
history = train_series.values.tolist()
forecasts = []

for start in range(0, len(test_series), block_size):
    remaining = len(test_series) - start
    current_horizon = min(horizon, remaining)
    # Forecast current block
    block_preds = []
    for step in range(current_horizon):
        # last `lags` values from history
        recent = history[-lags:]
        features = np.array(recent[::-1]).reshape(1, -1)  # lag_1 to lag_14 in correct order
        pred = model.predict(features)[0]
        block_preds.append(pred)
        history.append(pred)  # use predicted as lag for subsequent steps
    # Remove predicted values from history and replace with true values
    history = history[:-current_horizon]  # remove the predictions we just added
    # Get true values for this block
    true_block = test_series.iloc[start:start+current_horizon].values.tolist()
    history.extend(true_block)
    forecasts.extend(block_preds)  # keep forecasted values
    # Retrain on expanded history
    if len(test_series) - (start + current_horizon) > 0:
        expanded_series = pd.Series(history, index=range(len(history)))
        X_exp, y_exp = create_lag_features(expanded_series, lags)
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            objective='reg:squarederror',
            verbosity=0
        )
        model.fit(X_exp, y_exp)

print(forecasts)