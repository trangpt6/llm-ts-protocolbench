import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

# set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

def create_lagged_dataset(series, end_train_idx, lags=7, horizon=7):
    # series: full pandas Series
    # end_train_idx: last index (inclusive) allowed for the last target value
    # returns X (2D), y (2D) where y shape (n_samples, horizon)
    X_list, y_list = [], []
    # min_t is lags because we need lags previous values
    # max_t is such that t+horizon-1 <= end_train_idx, so t <= end_train_idx - horizon + 1
    min_t = lags
    max_t = end_train_idx - horizon + 1
    for t in range(min_t, max_t + 1):
        # features: series[t-lags : t-1] inclusive? indices from t-lags to t-1
        X_list.append(series.iloc[t-lags:t].values)
        # target: series[t:t+horizon] (t to t+horizon-1)
        y_list.append(series.iloc[t:t+horizon].values)
    return np.array(X_list), np.array(y_list)

# read data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.sort_values('Date', inplace=True)
series = df['Daily minimum temperatures']

# split sizes
train_size = 2921  # 0.8*3652 = 2921.6, int(0.8*3652)=2921
total = len(series)

# forecasting parameters
lags = 7
horizon = 7

# model hyperparameters
model_params = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbose': -1
}

forecasts = []

# rolling forecast
for i in range(train_size, total):
    # i is the index of the first test point we will forecast
    end_train = i - 1  # we can use all data up to i-1 for training
    # prepare training data up to end_train
    X_train, y_train = create_lagged_dataset(series, end_train, lags, horizon)
    # train multi-output model
    base_lgb = LGBMRegressor(**model_params)
    model = MultiOutputRegressor(base_lgb)
    model.fit(X_train, y_train)
    # prepare prediction input: last 'lags' values before i
    X_pred = series.iloc[i-lags:i].values.reshape(1, -1)
    y_pred = model.predict(X_pred)  # shape (1, horizon)
    # first step ahead forecast corresponds to index i
    forecasts.append(y_pred[0, 0])
    # Note: ground truth at i will be available in next iteration because we use series (which includes test values)
    # The series object contains the whole dataset; end_train in next iteration will be i, including the true value at i.

# convert to list
forecasts = list(forecasts)
print(forecasts)