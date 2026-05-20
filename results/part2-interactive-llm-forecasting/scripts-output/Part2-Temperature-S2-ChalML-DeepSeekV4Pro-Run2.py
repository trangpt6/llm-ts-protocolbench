import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# Reproducibility seeds
random.seed(42)
np.random.seed(42)

# Load data, extract target series
df = pd.read_csv(r'../../../data/Temperature.csv')
y = df['Daily minimum temperatures'].values.astype(float)

# Chronological split
train_size = 2921
test_start = train_size
test_count = len(y) - train_size

lags = 7
model_params = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbosity': -1
}

forecasts = []

for step in range(test_count):
    pred_idx = test_start + step

    # Features for prediction (most recent 7 values before pred_idx)
    X_pred = y[pred_idx - lags:pred_idx].reshape(1, -1)

    # Build training set from all available history up to pred_idx-1
    train_end = pred_idx
    if train_end > lags:
        X_train = np.array([y[i - lags:i] for i in range(lags, train_end)])
        y_train = y[lags:train_end]
    else:
        X_train = np.empty((0, lags))
        y_train = np.empty((0,))

    # Retrain model from scratch on all available data
    model = lgb.LGBMRegressor(**model_params)
    model.fit(X_train, y_train)

    pred = model.predict(X_pred)[0]
    forecasts.append(float(pred))

print(forecasts)