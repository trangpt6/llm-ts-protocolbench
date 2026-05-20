import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]

total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

lags = 6
forecasts = []

def make_supervised(data, lags, feature_cols, target_col):
    X, y = [], []
    values = data[feature_cols].to_numpy(dtype=float)
    target = data[target_col].to_numpy(dtype=float)
    for idx in range(lags, len(data)):
        X.append(values[idx - lags:idx].flatten())
        y.append(target[idx])
    return np.array(X), np.array(y)

for test_idx in range(train_size, total_timesteps):
    history = df.iloc[:test_idx].copy()
    X_train, y_train = make_supervised(history, lags, feature_cols, target_col)
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    X_pred = history[feature_cols].to_numpy(dtype=float)[-lags:].flatten().reshape(1, -1)
    pred = float(model.predict(X_pred)[0])
    forecasts.append(pred)

print(forecasts)