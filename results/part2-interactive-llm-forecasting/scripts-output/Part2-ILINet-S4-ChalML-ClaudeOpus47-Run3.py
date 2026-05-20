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

y = df['% WEIGHTED ILI'].values.astype(float)

n_total = len(y)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

LAGS = 52
BLOCK = 52

def build_features(series, lags, h):
    n = len(series)
    rows = []
    targets = []
    for t in range(lags, n - h + 1):
        rows.append(series[t - lags:t])
        targets.append(series[t + h - 1])
    return np.array(rows), np.array(targets)

forecasts = []
train_data = list(y[:n_train])
test_data = list(y[n_train:])

i = 0
while i < n_test:
    current_block_size = min(BLOCK, n_test - i)
    series = np.array(train_data)
    last_window = series[-LAGS:].reshape(1, -1)

    for h in range(1, current_block_size + 1):
        X_tr, y_tr = build_features(series, LAGS, h)
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbosity=0
        )
        model.fit(X_tr, y_tr)
        pred = float(model.predict(last_window)[0])
        forecasts.append(pred)

    train_data.extend(test_data[i:i + current_block_size])
    i += current_block_size

print(forecasts)