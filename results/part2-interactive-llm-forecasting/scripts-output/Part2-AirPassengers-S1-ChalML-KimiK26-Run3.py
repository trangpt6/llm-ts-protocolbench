import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset from local file
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# extract target series in chronological order
y = df['Passengers'].astype(float).values

# chronological 80/20 split
n_total = len(y)
train_size = int(0.8 * n_total)
train_data = y[:train_size]
test_size = n_total - train_size

# feature engineering using lag features
lags = 12
X_train = []
y_train = []
for i in range(lags, len(train_data)):
    X_train.append(train_data[i-lags:i][::-1])
    y_train.append(train_data[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# initialize and fit LightGBM with fixed hyperparameters
model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# recursive one-step-ahead forecasting over test set
history = list(train_data)
forecasts = []
for i in range(test_size):
    x_input = np.array(history[-lags:][::-1]).reshape(1, -1)
    pred = model.predict(x_input)[0]
    forecasts.append(float(pred))
    history.append(pred)

print(forecasts)