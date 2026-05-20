import pandas as pd
import numpy as np
import random
import xgboost as xgb

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

y = df['% WEIGHTED ILI'].astype(float).values

n = len(y)
train_size = 1040
test_size = n - train_size

lags = 52

def build_lag_matrix(series, lags):
    X = []
    Y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        Y.append(series[i])
    return np.array(X), np.array(Y)

train_series = y[:train_size]
X_train, y_train = build_lag_matrix(train_series, lags)

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=1,
    verbosity=0
)
model.fit(X_train, y_train)

history = list(y[train_size - lags:train_size])
forecasts = []
for _ in range(test_size):
    x_input = np.array(history[-lags:]).reshape(1, -1)
    pred = float(model.predict(x_input)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)