import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)

# read the dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
series = df['Passengers'].values.astype(float)

# split point: 115 training (0 to 114 index), 29 test (115 to 143 index)
train_size = 115   # 80% of 144

# hyperparameters
lags = 12
n_estimators = 50
max_depth = 4

forecasts = []

# rolling forecast with retraining at each step
for i in range(train_size, len(series)):
    # data available up to i-1
    hist = series[:i]
    # create lag features and targets from hist
    X_train, y_train = [], []
    for t in range(lags, len(hist)):
        X_train.append(hist[t-lags:t])
        y_train.append(hist[t])
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # train RandomForest
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)

    # prepare input for next step (last lags values of hist)
    last_lags = hist[-lags:]
    X_pred = last_lags.reshape(1, -1)
    y_pred = model.predict(X_pred)[0]
    forecasts.append(y_pred)

# print the forecast list (29 values)
print(forecasts)