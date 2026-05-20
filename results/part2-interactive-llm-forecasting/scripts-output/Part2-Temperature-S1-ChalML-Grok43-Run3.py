import pandas as pd
import numpy as np
import xgboost as xgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target_col = 'Daily minimum temperatures'
train = df.iloc[:2920].copy()
non_zero_median = train[train[target_col] != 0][target_col].median()
df[target_col] = df[target_col].replace(0, non_zero_median)
series = df[target_col].values
lags = 14
train_series = series[:2920]
X_train = []
y_train = []
for i in range(lags, len(train_series)):
    X_train.append(train_series[i-lags:i])
    y_train.append(train_series[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
history = list(train_series[-lags:])
forecasts = []
for _ in range(730):
    X_pred = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    history.append(pred)
print(forecasts)