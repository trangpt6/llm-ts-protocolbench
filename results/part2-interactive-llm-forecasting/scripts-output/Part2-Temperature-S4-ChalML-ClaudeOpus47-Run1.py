import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

values = df['Daily minimum temperatures'].values.astype(float)

total = len(values)
train_size = int(0.8 * total)
test_size = total - train_size

train = values[:train_size].copy()
test = values[train_size:].copy()

lags = 14
block_size = 30

def build_lag_matrix(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

forecasts = []
history = train.copy()
n_blocks = (test_size + block_size - 1) // block_size

for b in range(n_blocks):
    start = b * block_size
    end = min(start + block_size, test_size)
    steps = end - start

    X_train, y_train = build_lag_matrix(history, lags)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=1,
        verbosity=0
    )
    model.fit(X_train, y_train)

    window = history[-lags:].astype(float).tolist()
    block_preds = []
    for s in range(steps):
        x_in = np.array(window[-lags:], dtype=float).reshape(1, -1)
        pred = float(model.predict(x_in)[0])
        block_preds.append(pred)
        window.append(pred)

    forecasts.extend(block_preds)

    history = np.concatenate([history, test[start:end]])

forecasts = [float(v) for v in forecasts]
print(forecasts)