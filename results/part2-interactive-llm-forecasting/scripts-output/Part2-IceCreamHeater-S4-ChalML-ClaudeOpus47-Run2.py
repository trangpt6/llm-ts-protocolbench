import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# reproducibility seeds
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

heater = df['Heater'].values.astype(float)
ice = df['Ice cream'].values.astype(float)

n = len(df)
train_size = int(0.8 * n)

lags = 12
horizon = 12
block_size = 12

forecasts = []
current_end = train_size

while current_end < n:
    steps_needed = min(block_size, n - current_end)
    block_forecasts = []
    for h in range(1, horizon + 1):
        X_train = []
        y_train = []
        for t in range(lags, current_end - h + 1):
            feat = np.concatenate([heater[t - lags:t], ice[t - lags:t]])
            X_train.append(feat)
            y_train.append(ice[t + h - 1])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        feat_pred = np.concatenate([heater[current_end - lags:current_end], ice[current_end - lags:current_end]]).reshape(1, -1)
        pred = model.predict(feat_pred)[0]
        block_forecasts.append(float(pred))
    forecasts.extend(block_forecasts[:steps_needed])
    current_end += steps_needed

print(forecasts)