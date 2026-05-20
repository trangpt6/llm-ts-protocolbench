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

target_col = 'Daily minimum temperatures'
values = df[target_col].values.astype(float)

n_total = len(values)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

lags = 14
block_size = 30

def make_lag_matrix(series, lags):
    X = []
    y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

history = list(values[:n_train])
test_values = values[n_train:]

forecasts = []
n_blocks = (n_test + block_size - 1) // block_size

for b in range(n_blocks):
    X_train, y_train = make_lag_matrix(np.array(history), lags)
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)

    current_block_size = min(block_size, n_test - b * block_size)
    window = list(history[-lags:])
    for step in range(current_block_size):
        x_input = np.array(window[-lags:]).reshape(1, -1)
        pred = float(model.predict(x_input)[0])
        forecasts.append(pred)
        window.append(pred)

    start_idx = b * block_size
    end_idx = start_idx + current_block_size
    history.extend(list(test_values[start_idx:end_idx]))

print(forecasts)