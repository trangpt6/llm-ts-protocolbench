import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset from local file
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# chronological split: first 80% train, remainder test
train_size = int(0.8 * len(y))
train_y = y[:train_size]

# build lag feature matrix using fixed lag count
lags = 12
X_train = []
y_train = []
for i in range(lags, len(train_y)):
    X_train.append(train_y[i - lags:i])
    y_train.append(train_y[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# initialize and train LightGBM with fixed hyperparameters
model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# recursive one-step-ahead forecast over full test set without ground truth updates
history = train_y[-lags:].tolist()
forecasts = []
for _ in range(len(y) - train_size):
    X_pred = np.array(history).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    history.pop(0)
    history.append(pred)

print(forecasts)