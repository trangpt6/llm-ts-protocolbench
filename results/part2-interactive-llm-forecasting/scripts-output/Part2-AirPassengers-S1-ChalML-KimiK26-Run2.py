import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read the dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df = df.sort_values(by='Month').reset_index(drop=True)
passengers = df['Passengers'].values

# define chronological train/test split
train_size = int(0.8 * len(passengers))
train_data = passengers[:train_size]
test_size = len(passengers) - train_size

# create lag features for training using fixed lag length
lags = 12
X_train = []
y_train = []
for i in range(lags, len(train_data)):
    X_train.append(train_data[i - lags:i])
    y_train.append(train_data[i])
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

# recursive one-step ahead forecasting over the test set
history = train_data[-lags:].tolist()
forecasts = []
for _ in range(test_size):
    X_input = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(X_input)[0]
    forecasts.append(float(pred))
    history.append(pred)

print(forecasts)