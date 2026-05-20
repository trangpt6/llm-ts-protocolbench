import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)
target_col = '% WEIGHTED ILI'

# replace zeros in target with NaN and interpolate linearly (as per Turn 1)
df[target_col] = df[target_col].replace(0.0, np.nan)
df[target_col] = df[target_col].interpolate(method='linear')

# chronological split: first 1044 rows train, rest 261 test
train_size = 1044
test_size = 261
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:train_size+test_size]

# full series for easy indexing
y_full = df[target_col].values

def create_lags(series, lags):
    n = len(series)
    X = []
    y = []
    for i in range(lags, n):
        X.append(series[i-lags:i].tolist())
        y.append(series[i])
    return np.array(X), np.array(y)

lags = 13
forecasts = []

# rolling process: for each test step
for i in range(test_size):
    # current training indices: 0 .. train_size + i - 1 (since i=0 -> last index 1043)
    train_end = train_size + i
    train_series = y_full[:train_end]   # true values only, no future
    X_train, y_train = create_lags(train_series, lags)
    
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # last lags from training series
    last_lags = list(train_series[-lags:])
    preds = []
    # recursive 4-step forecast
    for step in range(4):
        X_step = np.array(last_lags).reshape(1, -1)
        pred = model.predict(X_step)[0]
        preds.append(pred)
        # update lags for next step
        last_lags.pop(0)
        last_lags.append(pred)
    # first prediction is for the current test point
    forecasts.append(preds[0])
    
    # after prediction, the true value becomes available for next iteration
    # (implicitly used because next iteration will include this index)

print(forecasts)