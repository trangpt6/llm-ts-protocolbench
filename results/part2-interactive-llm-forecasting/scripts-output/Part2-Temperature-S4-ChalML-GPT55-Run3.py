import random
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Random seed setting.
random.seed(42)
np.random.seed(42)
warnings.filterwarnings("ignore")

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df["Daily minimum temperatures"] = pd.to_numeric(df["Daily minimum temperatures"])
df = df.sort_values("Date").reset_index(drop=True)

full_index = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
missing_dates = full_index.difference(df["Date"])

target_col = "Daily minimum temperatures"
lags = 14
train_size = 2920
test_size = 730
block_size = 30
horizon = 30

train_values = df.iloc[:train_size][target_col].to_numpy(dtype=float)
test_values = df.iloc[train_size:train_size + test_size][target_col].to_numpy(dtype=float)

def make_lagged_features(values, lags):
    values = np.asarray(values, dtype=float)
    X = []
    y = []
    for i in range(lags, len(values)):
        X.append(values[i - lags:i])
        y.append(values[i])
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

history = list(train_values)
forecasts = []

for start in range(0, len(test_values), block_size):
    X_train, y_train = make_lagged_features(history, lags)
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        verbosity=0
    )
    model.fit(X_train, y_train)

    steps = min(horizon, len(test_values) - start)
    prediction_history = list(history)
    block_forecasts = []

    for _ in range(steps):
        x_pred = np.asarray(prediction_history[-lags:], dtype=float).reshape(1, -1)
        y_pred = float(model.predict(x_pred)[0])
        block_forecasts.append(y_pred)
        prediction_history.append(y_pred)

    forecasts.extend(block_forecasts)
    history.extend(test_values[start:start + steps].tolist())

forecasts = forecasts[:test_size]
print(forecasts)