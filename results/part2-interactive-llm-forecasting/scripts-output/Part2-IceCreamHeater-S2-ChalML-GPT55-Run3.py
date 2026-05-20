import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]
lags = 6
train_size = 158
test_size = 40

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:train_size + test_size].copy()

def make_lagged_xy(data, feature_cols, target_col, lags):
    X = []
    y = []
    values = data[feature_cols].to_numpy(dtype=float)
    target = data[target_col].to_numpy(dtype=float)
    for i in range(lags, len(data)):
        row = []
        for lag in range(1, lags + 1):
            row.extend(values[i - lag])
        X.append(row)
        y.append(target[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def make_next_x(history, feature_cols, lags):
    values = history[feature_cols].to_numpy(dtype=float)
    row = []
    for lag in range(1, lags + 1):
        row.extend(values[-lag])
    return np.array([row], dtype=float)

history = train_df.copy()
forecasts = []

for i in range(len(test_df)):
    X_train, y_train = make_lagged_xy(history, feature_cols, target_col, lags)
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    X_next = make_next_x(history, feature_cols, lags)
    pred = float(model.predict(X_next)[0])
    forecasts.append(pred)
    history = pd.concat([history, test_df.iloc[[i]].copy()], ignore_index=True)

print(forecasts)