import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

target = "% WEIGHTED ILI"
y = df[target].astype(float).to_numpy()

train_size = 1040
test_size = 261
lags = 13
horizon = 4

train_y = y[:train_size]
test_y = y[train_size:train_size + test_size]

def make_lag_matrix(series, n_lags):
    X = []
    target_values = []
    for i in range(n_lags, len(series)):
        X.append(series[i - n_lags:i][::-1])
        target_values.append(series[i])
    return np.asarray(X, dtype=float), np.asarray(target_values, dtype=float)

history = list(train_y)
forecasts = []

for i in range(test_size):
    X_train, y_train = make_lag_matrix(np.asarray(history, dtype=float), lags)
    model = LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    recursive_history = list(history)
    step_forecasts = []
    for _ in range(horizon):
        x_input = np.asarray(recursive_history[-lags:][::-1], dtype=float).reshape(1, -1)
        y_pred = float(model.predict(x_input)[0])
        step_forecasts.append(y_pred)
        recursive_history.append(y_pred)

    forecasts.append(step_forecasts[0])
    history.append(float(test_y[i]))

print(forecasts)