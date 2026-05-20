import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]
lags = 12
train_size = 158
test_size = 40

train_df = df.iloc[:train_size].copy()
full_df = df.copy()

X_train = []
y_train = []
for i in range(lags, train_size):
    row = []
    for col in feature_cols:
        for lag in range(1, lags + 1):
            row.append(train_df[col].iloc[i - lag])
    X_train.append(row)
    y_train.append(train_df[target_col].iloc[i])

X_train = np.array(X_train, dtype=float)
y_train = np.array(y_train, dtype=float)

model = LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    deterministic=True,
    force_col_wise=True,
    verbose=-1
)
model.fit(X_train, y_train)

target_history = list(train_df[target_col].astype(float).values)
forecasts = []

for step in range(test_size):
    idx = train_size + step
    row = []
    for col in feature_cols:
        if col == target_col:
            history = target_history
            for lag in range(1, lags + 1):
                row.append(history[-lag])
        else:
            for lag in range(1, lags + 1):
                row.append(float(full_df[col].iloc[idx - lag]))
    X_pred = np.array(row, dtype=float).reshape(1, -1)
    pred = float(model.predict(X_pred)[0])
    forecasts.append(pred)
    target_history.append(pred)

print(forecasts)