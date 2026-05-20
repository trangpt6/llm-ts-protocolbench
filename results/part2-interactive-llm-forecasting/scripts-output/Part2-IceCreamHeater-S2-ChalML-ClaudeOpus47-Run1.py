import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

total = len(df)
train_size = int(0.8 * total)

lags = 6
features = ['Heater', 'Ice cream']
target = 'Ice cream'

def make_features(data, lags, features, target):
    X_list = []
    y_list = []
    for i in range(lags, len(data)):
        row = []
        for f in features:
            for l in range(1, lags + 1):
                row.append(data[f].iloc[i - l])
        X_list.append(row)
        y_list.append(data[target].iloc[i])
    return np.array(X_list), np.array(y_list)

forecasts = []
for t in range(train_size, total):
    history = df.iloc[:t].copy()
    X_train, y_train = make_features(history, lags, features, target)

    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    row = []
    for f in features:
        for l in range(1, lags + 1):
            row.append(history[f].iloc[len(history) - l])
    X_test = np.array(row, dtype=float).reshape(1, -1)
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))

print(forecasts)