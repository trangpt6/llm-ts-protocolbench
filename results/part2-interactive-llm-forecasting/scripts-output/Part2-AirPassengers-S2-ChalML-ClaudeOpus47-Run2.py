import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

values = df['Passengers'].values.astype(float)
n = len(values)
train_size = int(0.8 * n)
test_size = n - train_size

lags = 12

def make_features(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

forecasts = []
history = list(values[:train_size])

for step in range(test_size):
    arr = np.array(history)
    X_train, y_train = make_features(arr, lags)
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    x_input = np.array(history[-lags:]).reshape(1, -1)
    pred = float(model.predict(x_input)[0])
    forecasts.append(pred)
    true_val = float(values[train_size + step])
    history.append(true_val)

print(forecasts)