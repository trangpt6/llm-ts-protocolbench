import pandas as pd
import numpy as np
import xgboost as xgb
import random

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read the dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()

# define target and initial train size
target = 'Passengers'
series = df[target].values
total_len = len(series)
train_size = int(0.8 * total_len)  # 115

# fixed hyperparameters
lags = 12
n_estimators = 50
max_depth = 3
learning_rate = 0.1

def create_lag_features(data, lags):
    n = len(data)
    X = np.zeros((n - lags, lags))
    y = np.zeros(n - lags)
    for i in range(lags, n):
        X[i - lags] = data[i - lags:i]
        y[i - lags] = data[i]
    return X, y

# storage for forecasts
forecasts = []
# initial training series will be expanded with true values as we go
train_series = series[:train_size].copy()

# rolling one-step ahead forecasting over test period
for t in range(train_size, total_len):
    # create features from current train_series
    X_train, y_train = create_lag_features(train_series, lags)
    # train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X_train, y_train)
    # prepare the last lags from train_series for prediction
    last_window = train_series[-lags:].reshape(1, -1)
    # predict next step
    pred = model.predict(last_window)[0]
    forecasts.append(pred)
    # append true value from test set to train_series for next iteration
    train_series = np.append(train_series, series[t])

print(forecasts)