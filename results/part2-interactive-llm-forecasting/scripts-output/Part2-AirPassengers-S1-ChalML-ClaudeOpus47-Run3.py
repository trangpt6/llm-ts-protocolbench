import random
import numpy as np
import pandas as pd
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

values = df['Passengers'].astype(float).values
n = len(values)
train_size = int(0.8 * n)
train = values[:train_size]
test = values[train_size:]

lags = 12
X_train = []
y_train = []
for i in range(lags, len(train)):
    X_train.append(train[i-lags:i])
    y_train.append(train[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

history = list(train[-lags:])
forecasts = []
for _ in range(len(test)):
    x_input = np.array(history[-lags:]).reshape(1, -1)
    pred = float(model.predict(x_input)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)