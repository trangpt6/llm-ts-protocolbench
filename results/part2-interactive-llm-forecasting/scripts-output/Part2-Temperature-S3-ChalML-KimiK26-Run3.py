import random
import numpy as np
import pandas as pd
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
y = df['Daily minimum temperatures'].to_numpy()

train_size = 2920
test_size = 730
train = y[:train_size]
test = y[train_size:train_size + test_size]

lags = 7
horizon = 7
history = train.tolist()
forecasts = []

for i in range(test_size):
    hist_arr = np.array(history)
    n = len(hist_arr)
    windows = np.lib.stride_tricks.sliding_window_view(hist_arr, lags)
    current_X = windows[-1].reshape(1, -1)
    for h in range(1, horizon + 1):
        max_i = n - h - lags
        X_h = windows[:max_i + 1]
        y_h = hist_arr[lags - 1 + h:n]
        model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(X_h, y_h)
        pred = model.predict(current_X)[0]
        forecasts.append(float(pred))
    history.append(float(test[i]))

print(forecasts)