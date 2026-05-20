import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)

target_col = "Daily minimum temperatures"
train_size = 2920
test_size = 730

train = df.iloc[:train_size].copy()
test = df.iloc[train_size:train_size + test_size].copy()

lags = 7
history = train[target_col].astype(float).tolist()
test_values = test[target_col].astype(float).tolist()
forecasts = []

def make_lag_matrix(values, lags):
    X = []
    y = []
    for i in range(lags, len(values)):
        X.append(values[i - lags:i][::-1])
        y.append(values[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)

for true_value in test_values:
    X_train, y_train = make_lag_matrix(history, lags)
    model = LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    x_pred = np.array(history[-lags:][::-1], dtype=float).reshape(1, -1)
    pred = float(model.predict(x_pred)[0])
    forecasts.append(pred)
    history.append(float(true_value))

print(forecasts)