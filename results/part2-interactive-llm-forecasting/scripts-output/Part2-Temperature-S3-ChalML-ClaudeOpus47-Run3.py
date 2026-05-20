import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
y = df['Daily minimum temperatures'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
LAGS = 7
H = 7

forecasts = []
n_windows = n - train_size - H + 1

for w in range(n_windows):
    L = train_size + w
    series = y[:L]
    last_feats = series[L - LAGS:L].reshape(1, -1)
    windows = np.lib.stride_tricks.sliding_window_view(series, LAGS)
    for h in range(1, H + 1):
        n_samples = L - LAGS + 1 - h
        X_train = windows[:n_samples]
        y_train = series[LAGS - 1 + h: LAGS - 1 + h + n_samples]
        model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        pred = float(model.predict(last_feats)[0])
        forecasts.append(pred)

print(forecasts)