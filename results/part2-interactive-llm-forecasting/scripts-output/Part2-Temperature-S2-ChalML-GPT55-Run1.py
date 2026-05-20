import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

target_col = "Daily minimum temperatures"
y = df[target_col].astype(float).to_numpy()

train_size = 2920
test_size = 730

train_y = y[:train_size]
test_y = y[train_size:train_size + test_size]

lags = 7
forecasts = []
history = train_y.copy()

def make_lag_features(values, lags):
    X = []
    y_out = []
    for i in range(lags, len(values)):
        X.append(values[i - lags:i][::-1])
        y_out.append(values[i])
    return np.asarray(X), np.asarray(y_out)

for i in range(test_size):
    X_train, y_train = make_lag_features(history, lags)
    model = LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    X_pred = history[-lags:][::-1].reshape(1, -1)
    pred = float(model.predict(X_pred)[0])
    forecasts.append(pred)
    history = np.append(history, test_y[i])

print(forecasts)