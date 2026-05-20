import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

y = df['Daily minimum temperatures'].astype(float).values
n = len(y)
train_size = int(0.8 * n)
y_train = y[:train_size]
y_test = y[train_size:]

lags = 14
X_train = np.zeros((len(y_train) - lags, lags))
t_train = np.zeros(len(y_train) - lags)
for i in range(lags, len(y_train)):
    X_train[i - lags] = y_train[i - lags:i]
    t_train[i - lags] = y_train[i]

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
)
model.fit(X_train, t_train)

window = list(y_train[-lags:])
forecasts = []
for _ in range(len(y_test)):
    x_in = np.array(window[-lags:]).reshape(1, -1)
    pred = float(model.predict(x_in)[0])
    forecasts.append(pred)
    window.append(pred)

print(forecasts)