import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.sort_values('Month').reset_index(drop=True)

target_col = 'Ice cream'
feat_cols = ['Heater', 'Ice cream']
lags = 12
horizon = 12
block_size = 12

n = len(df)
train_size = int(0.8 * n)
test_size = n - train_size

series_2d = df[feat_cols].values.astype(float)
target_arr = df[target_col].values.astype(float)

def build_direct_dataset(end_idx_inclusive, h):
    X_list, y_list = [], []
    for i in range(lags - 1, end_idx_inclusive - h + 1):
        feat = series_2d[i - lags + 1:i + 1].flatten()
        X_list.append(feat)
        y_list.append(target_arr[i + h])
    return np.array(X_list), np.array(y_list)

forecasts = []
origin = train_size - 1

while len(forecasts) < test_size:
    steps_this_block = min(block_size, test_size - len(forecasts))
    feat_now = series_2d[origin - lags + 1:origin + 1].flatten().reshape(1, -1)
    for h in range(1, steps_this_block + 1):
        X_train, y_train = build_direct_dataset(origin, h)
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(feat_now)[0])
        forecasts.append(pred)
    origin += steps_this_block

print(forecasts)