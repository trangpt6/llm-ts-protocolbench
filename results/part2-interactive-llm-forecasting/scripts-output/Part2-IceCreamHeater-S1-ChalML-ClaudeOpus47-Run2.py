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

n = len(df)
train_size = int(0.8 * n)
lags = 12

heater = df['Heater'].values.astype(float)
icecream = df['Ice cream'].values.astype(float)

X_train = []
y_train = []
for i in range(lags, train_size):
    feat = list(heater[i-lags:i]) + list(icecream[i-lags:i])
    X_train.append(feat)
    y_train.append(icecream[i])

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

icecream_dyn = icecream.copy()
forecasts = []
for i in range(train_size, n):
    feat = np.array(list(heater[i-lags:i]) + list(icecream_dyn[i-lags:i])).reshape(1, -1)
    pred = float(model.predict(feat)[0])
    forecasts.append(pred)
    icecream_dyn[i] = pred

print(forecasts)