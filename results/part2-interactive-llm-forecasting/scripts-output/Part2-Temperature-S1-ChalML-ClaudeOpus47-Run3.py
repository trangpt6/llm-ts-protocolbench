import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

target_col = 'Daily minimum temperatures'
y = df[target_col].astype(float).values

n_total = len(y)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

lags = 14

train_y = y[:n_train]

X_train = []
y_train = []
for i in range(lags, n_train):
    X_train.append(train_y[i - lags:i])
    y_train.append(train_y[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=1,
)
model.fit(X_train, y_train)

history = list(train_y)
forecasts = []
for _ in range(n_test):
    window = np.array(history[-lags:]).reshape(1, -1)
    pred = float(model.predict(window)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)