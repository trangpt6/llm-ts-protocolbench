import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

target_col = "% WEIGHTED ILI"
lags = 52
train_size = 1040
test_size = 261

y = df[target_col].astype(float).to_numpy()
train_y = y[:train_size]

X_train = []
y_train = []
for i in range(lags, train_size):
    X_train.append(train_y[i - lags:i])
    y_train.append(train_y[i])

X_train = np.asarray(X_train, dtype=float)
y_train = np.asarray(y_train, dtype=float)

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=0,
    objective="reg:squarederror",
    verbosity=0
)
model.fit(X_train, y_train)

history = list(train_y)
forecasts = []
for _ in range(test_size):
    x_input = np.asarray(history[-lags:], dtype=float).reshape(1, -1)
    pred = float(model.predict(x_input)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)