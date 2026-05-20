import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
temps = df['Daily minimum temperatures'].astype(float).values
for i in range(len(temps)):
    if temps[i] == 0:
        preceding = []
        j = i - 1
        while len(preceding) < 7 and j >= 0:
            if temps[j] != 0:
                preceding.append(temps[j])
            j -= 1
        if preceding:
            temps[i] = np.median(preceding)
df['Daily minimum temperatures'] = temps
train_size = 2920
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()
models = {}
for h in range(1, 8):
    train_y = train_df['Daily minimum temperatures'].values
    X_train = []
    y_train_h = []
    for i in range(7, len(train_y) - h + 1):
        X_train.append(train_y[i-7:i])
        y_train_h.append(train_y[i + h - 1])
    X_train = np.array(X_train)
    y_train_h = np.array(y_train_h)
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train_h)
    models[h] = model
forecasts = []
current_train = train_df['Daily minimum temperatures'].values.tolist()
for i in range(len(test_df)):
    last_7 = np.array(current_train[-7:]).reshape(1, -1)
    pred = models[1].predict(last_7)[0]
    forecasts.append(pred)
    true_value = test_df['Daily minimum temperatures'].iloc[i]
    current_train.append(true_value)
    for h in range(1, 8):
        X_new = []
        y_new = []
        for j in range(7, len(current_train) - h + 1):
            X_new.append(current_train[j-7:j])
            y_new.append(current_train[j + h - 1])
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        new_model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        new_model.fit(X_new, y_new)
        models[h] = new_model
print(forecasts)