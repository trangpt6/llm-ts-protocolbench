import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)

full_index = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
regularized = df.set_index("Date").reindex(full_index)
regularized.index.name = "Date"

observed = regularized.dropna(subset=["Daily minimum temperatures"]).reset_index()
observed = observed.rename(columns={"index": "Date"})

train_size = 2920
test_size = 730
target_col = "Daily minimum temperatures"
lags = 7
horizon = 7

train_values = observed.iloc[:train_size][target_col].astype(float).to_numpy()
test_values = observed.iloc[train_size:train_size + test_size][target_col].astype(float).to_numpy()

def make_direct_training_data(values, lags, horizon):
    X = []
    Y = []
    n = len(values)
    for t in range(lags, n - horizon + 1):
        X.append(values[t - lags:t][::-1])
        Y.append(values[t:t + horizon])
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

forecasts = []
history = train_values.copy()

for i in range(test_size):
    X_train, Y_train = make_direct_training_data(history, lags, horizon)
    x_pred = history[-lags:][::-1].reshape(1, -1)
    step_predictions = []
    for h in range(horizon):
        model = LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_train, Y_train[:, h])
        step_predictions.append(float(model.predict(x_pred)[0]))
    forecasts.append(step_predictions[0])
    history = np.append(history, test_values[i])

print(forecasts)