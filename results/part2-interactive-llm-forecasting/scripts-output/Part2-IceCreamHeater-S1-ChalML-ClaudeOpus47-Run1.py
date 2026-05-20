import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

feature_cols = ['Heater', 'Ice cream']
target_idx = 1
lags = 12

data = df[feature_cols].values.astype(float)
n = len(df)
train_size = int(0.8 * n)

def build_lag_matrix(arr, lags):
    X, y = [], []
    for i in range(lags, len(arr)):
        row = []
        for l in range(1, lags + 1):
            row.extend(arr[i - l])
        X.append(row)
        y.append(arr[i, target_idx])
    return np.array(X), np.array(y)

train_arr = data[:train_size]
X_train, y_train = build_lag_matrix(train_arr, lags)

model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)

history = data[:train_size].copy()
forecasts = []

for t in range(train_size, n):
    row = []
    for l in range(1, lags + 1):
        row.extend(history[-l])
    X_test = np.array(row).reshape(1, -1)
    pred = float(model.predict(X_test)[0])
    forecasts.append(pred)
    new_row = np.array([data[t, 0], pred])
    history = np.vstack([history, new_row])

print(forecasts)