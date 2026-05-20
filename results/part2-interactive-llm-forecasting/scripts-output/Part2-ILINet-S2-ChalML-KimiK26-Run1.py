import random
import numpy as np
import pandas as pd
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')

df['AGE 25-49'] = df['AGE 25-49'].replace('', np.nan)
df['AGE 50-64'] = df['AGE 50-64'].replace('', np.nan)

if 'AGE 25-64' in df.columns:
    df = df.drop(columns=['AGE 25-64'])

y = df['% WEIGHTED ILI'].astype(float).to_numpy()
n_total = len(y)
train_size = int(0.8 * n_total)
lags = 13
n_estimators = 50
max_depth = 4
learning_rate = 0.1

forecasts = []

for t in range(train_size, n_total):
    train_indices = np.arange(lags, t)
    X_train = np.zeros((len(train_indices), lags), dtype=float)
    for idx, i in enumerate(train_indices):
        X_train[idx, :] = y[i - lags:i][::-1]
    y_train = y[lags:t]

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)

    X_pred = y[t - lags:t][::-1].reshape(1, -1)
    y_hat = model.predict(X_pred)[0]
    forecasts.append(float(y_hat))

print(forecasts)