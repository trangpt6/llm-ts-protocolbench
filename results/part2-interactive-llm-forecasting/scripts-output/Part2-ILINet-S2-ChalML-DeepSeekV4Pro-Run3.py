import pandas as pd
import numpy as np
import lightgbm as lgb

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'], index_col='DATE')

# Drop fully empty columns as per Turn 1 preprocessing
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True, errors='ignore')

# Define target column
target_col = '% WEIGHTED ILI'
series = df[target_col].astype(float).values

# Chronological split
train_size = 1053
test_size = len(series) - train_size
assert test_size == 264

def create_lagged_features(y, lags=13):
    X, Y = [], []
    for i in range(lags, len(y)):
        X.append(y[i-lags:i][::-1])
        Y.append(y[i])
    return np.array(X), np.array(Y)

# Fixed hyperparameters
params = {
    'n_estimators': 50,
    'max_depth': 4,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbosity': -1
}

# Initial training on first train_size observations
history = series.copy()
X_train, y_train = create_lagged_features(history[:train_size])
if len(X_train) > 0:
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
else:
    raise ValueError('Not enough training samples after lag creation')

# Rolling one-step ahead forecasts
forecasts = []
for t in range(train_size, len(series)):
    # Build lag input from history up to t
    past = history[t-13:t]
    X_pred = past[::-1].reshape(1, -1)
    y_hat = model.predict(X_pred)[0]
    forecasts.append(float(y_hat))
    # Update model with true value
    true_val = history[t]
    # Retrain on all data up to t (now t is known)
    X_new_train, y_new_train = create_lagged_features(history[:t+1])
    if len(X_new_train) > 0:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_new_train, y_new_train)

print(forecasts)