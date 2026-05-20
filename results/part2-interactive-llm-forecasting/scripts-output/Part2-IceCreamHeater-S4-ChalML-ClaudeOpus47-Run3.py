import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']

n = len(df)
train_size = int(0.8 * n)
test_size = n - train_size

lags = 12
horizon = 12
block_size = 12

all_values = df[feature_cols].values.astype(float)
target_idx = feature_cols.index(target_col)

def make_xy(values, lags, horizon_step, target_idx):
    n_samples = values.shape[0]
    X_list = []
    y_list = []
    for i in range(lags - 1, n_samples - horizon_step):
        X_list.append(values[i - lags + 1:i + 1, :].flatten())
        y_list.append(values[i + horizon_step, target_idx])
    return np.array(X_list), np.array(y_list)

forecasts = []
block_start = train_size

while block_start < n:
    current_block_size = min(block_size, n - block_start)
    train_values = all_values[:block_start]

    models = {}
    for h in range(1, horizon + 1):
        X_train, y_train = make_xy(train_values, lags, h, target_idx)
        model = LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        models[h] = model

    X_input = train_values[-lags:, :].flatten().reshape(1, -1)

    for h in range(1, current_block_size + 1):
        pred = models[h].predict(X_input)[0]
        forecasts.append(float(pred))

    block_start += current_block_size

print(forecasts)