import random
random.seed(42)
import numpy as np
np.random.seed(42)
import pandas as pd
import lightgbm as lgb

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df.sort_values('Month', inplace=True)
df.reset_index(drop=True, inplace=True)

train_size = 158
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

lags = 12
horizon = 12
n_estimators = 200
max_depth = 5
learning_rate = 0.05

def create_features_targets(history_df):
    Heater = history_df['Heater'].values
    IceCream = history_df['Ice cream'].values
    X = []
    Y = {h: [] for h in range(1, horizon + 1)}
    n = len(history_df)
    for i in range(lags, n - horizon + 1):
        feat_heater = Heater[i - lags:i]
        feat_ice = IceCream[i - lags:i]
        features = np.concatenate([feat_heater, feat_ice])
        X.append(features)
        for h in range(1, horizon + 1):
            Y[h].append(IceCream[i + h - 1])
    X = np.array(X)
    for h in range(1, horizon + 1):
        Y[h] = np.array(Y[h])
    return X, Y

history = train_df.copy()
X_train, Y_train = create_features_targets(history)
models = {}
for h in range(1, horizon + 1):
    model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, verbose=-1)
    model.fit(X_train, Y_train[h])
    models[h] = model

forecasts_list = []
current_history = history.copy()
block_start = 0
while True:
    block_end = min(block_start + horizon, len(test_df))
    needed_steps = block_end - block_start
    last_12 = current_history.iloc[-lags:]
    latest_heater = last_12['Heater'].values
    latest_ice = last_12['Ice cream'].values
    X_pred = np.concatenate([latest_heater, latest_ice]).reshape(1, -1)
    preds = []
    for h in range(1, needed_steps + 1):
        y_pred = models[h].predict(X_pred)[0]
        preds.append(y_pred)
    forecasts_list.extend(preds)
    if block_end == len(test_df):
        break
    true_block = test_df.iloc[block_start:block_end]
    current_history = pd.concat([current_history, true_block], ignore_index=True)
    X_train, Y_train = create_features_targets(current_history)
    for h in range(1, horizon + 1):
        model = lgb.LGBMRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42, verbose=-1)
        model.fit(X_train, Y_train[h])
        models[h] = model
    block_start = block_end

print(forecasts_list)