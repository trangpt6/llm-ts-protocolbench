import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
exog_col = "Heater"
train_size = 158
test_size = 40
lags = 6
horizon = 12

def make_supervised(data, end_idx):
    X = []
    y = []
    for i in range(lags, end_idx):
        features = []
        for lag in range(1, lags + 1):
            features.append(data[target_col].iloc[i - lag])
        for lag in range(1, lags + 1):
            features.append(data[exog_col].iloc[i - lag])
        X.append(features)
        y.append(data[target_col].iloc[i])
    return np.array(X), np.array(y)

forecasts = []

for step in range(test_size):
    current_end = train_size + step
    X_train, y_train = make_supervised(df, current_end)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_history = list(df[target_col].iloc[:current_end].values)
    exog_history = list(df[exog_col].iloc[:current_end].values)
    recursive_preds = []

    for h in range(horizon):
        features = []
        for lag in range(1, lags + 1):
            features.append(y_history[-lag])
        for lag in range(1, lags + 1):
            features.append(exog_history[-lag])
        pred = float(model.predict(np.array(features).reshape(1, -1))[0])
        recursive_preds.append(pred)
        y_history.append(pred)
        future_idx = current_end + h
        if future_idx < len(df):
            exog_history.append(df[exog_col].iloc[future_idx])
        else:
            exog_history.append(exog_history[-1])

    forecasts.append(recursive_preds[0])

print(forecasts)