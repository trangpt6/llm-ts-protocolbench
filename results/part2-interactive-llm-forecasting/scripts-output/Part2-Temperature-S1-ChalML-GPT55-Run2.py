import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)

target_col = "Daily minimum temperatures"
values = df[target_col].astype(float).to_numpy()

train_size = 2920
test_size = 730
lags = 14

train_values = values[:train_size]

X_train = []
y_train = []
 for_i = range(lags, train_size)
for i in for_i:
    X_train.append(train_values[i - lags:i])
    y_train.append(train_values[i])

X_train = np.asarray(X_train, dtype=float)
y_train = np.asarray(y_train, dtype=float)

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=1
)
model.fit(X_train, y_train)

history = list(train_values)
forecasts = []

for _ in range(test_size):
    x = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
    pred = float(model.predict(x)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)