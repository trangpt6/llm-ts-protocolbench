import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
df = df.set_index('Date')
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df = df.reindex(full_idx)
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='linear')

series = df['Daily minimum temperatures'].values.astype(float)

total = len(series)
train_size = int(0.8 * total)
train = series[:train_size]
test = series[train_size:]

lags = 14

X_train = []
y_train = []
for i in range(lags, len(train)):
    X_train.append(train[i - lags:i])
    y_train.append(train[i])
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

history = list(train[-lags:])
forecasts = []
for _ in range(len(test)):
    x = np.array(history[-lags:]).reshape(1, -1)
    yhat = float(model.predict(x)[0])
    forecasts.append(yhat)
    history.append(yhat)

forecasts = [float(v) for v in forecasts]
print(forecasts)