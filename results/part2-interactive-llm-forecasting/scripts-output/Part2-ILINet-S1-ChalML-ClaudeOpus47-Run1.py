import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'
y = df[target_col].astype(float).copy()

mask = (df['DATE'] < pd.Timestamp('2003-10-05')) & (y == 0.0)
y[mask] = np.nan
y = y.interpolate(method='linear', limit_direction='both')

values = y.values.astype(float)

n_total = len(values)
n_train = int(0.8 * n_total)
n_test = n_total - n_train
lags = 52

X_train = []
y_train = []
for i in range(lags, n_train):
    X_train.append(values[i - lags:i])
    y_train.append(values[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    n_jobs=1,
    verbosity=0
)
model.fit(X_train, y_train)

history = list(values[:n_train])
forecasts = []
for _ in range(n_test):
    x = np.array(history[-lags:], dtype=float).reshape(1, -1)
    pred = float(model.predict(x)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)