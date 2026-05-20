import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
df.columns = ['Date', 'Daily minimum temperatures']
df['Daily minimum temperatures'] = pd.to_numeric(df['Daily minimum temperatures'], errors='coerce')
target_col = 'Daily minimum temperatures'
zero_indices = df[df[target_col] == 0].index.tolist()
for idx in zero_indices:
    if idx >= 7:
        median_val = df[target_col].iloc[idx-7:idx].median()
        df.loc[idx, target_col] = median_val
    else:
        df.loc[idx, target_col] = df[target_col].iloc[:idx].median() if idx > 0 else 0
train_size = 2920
train = df.iloc[:train_size][target_col].values
test = df.iloc[train_size:][target_col].values
forecasts = []
current_train = list(train)
for step in range(len(test)):
    X_list = []
    y_list = [[] for _ in range(7)]
    for i in range(7, len(current_train)):
        X_list.append(current_train[i-7:i])
        for h in range(7):
            if i + h < len(current_train):
                y_list[h].append(current_train[i + h])
    if len(X_list) == 0:
        pred = np.mean(current_train[-7:])
        forecasts.append(pred)
    else:
        X = np.array(X_list)
        preds = []
        for h in range(7):
            y_h = np.array(y_list[h])
            if len(y_h) > 0:
                model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
                model.fit(X[:len(y_h)], y_h)
                last_X = np.array(current_train[-7:]).reshape(1, -1)
                pred_h = model.predict(last_X)[0]
                preds.append(pred_h)
            else:
                preds.append(np.mean(current_train[-7:]))
        forecasts.append(preds[0])
    current_train.append(test[step])
print(forecasts)