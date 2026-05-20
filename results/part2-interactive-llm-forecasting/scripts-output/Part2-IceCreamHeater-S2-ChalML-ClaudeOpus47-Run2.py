import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]

lags = 6
n_total = len(df)
n_train = int(0.8 * n_total)

values = df[feature_cols].values.astype(float)
target = df[target_col].values.astype(float)

def build_xy(end_idx):
    X, y = [], []
    for t in range(lags, end_idx):
        feats = []
        for c in range(values.shape[1]):
            feats.extend(values[t - lags:t, c].tolist())
        X.append(feats)
        y.append(target[t])
    return np.array(X), np.array(y)

forecasts = []
for step in range(n_train, n_total):
    X_train, y_train = build_xy(step)
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=0,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    feats = []
    for c in range(values.shape[1]):
        feats.extend(values[step - lags:step, c].tolist())
    x_pred = np.array(feats).reshape(1, -1)
    yhat = float(model.predict(x_pred)[0])
    forecasts.append(yhat)

print(forecasts)