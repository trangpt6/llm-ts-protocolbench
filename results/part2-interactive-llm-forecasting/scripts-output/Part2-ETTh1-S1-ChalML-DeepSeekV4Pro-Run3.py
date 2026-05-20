import random
import numpy as np
import pandas as pd
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
y = df['OT'].values

# Split
train_size = 14016
train_y = y[:train_size]
test_y = y[train_size:]
test_len = len(test_y)

# Build supervised data for training with lags=24
lag = 24
X_train = []
y_train = []
for i in range(lag, len(train_y)):
    X_train.append(train_y[i-lag:i])
    y_train.append(train_y[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Train model
model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)

# Recursive forecasting
# Use last 'lag' values from training as initial history
history = train_y[-lag:].tolist()
forecasts = []

for _ in range(test_len):
    X_input = np.array(history[-lag:]).reshape(1, -1)
    pred = model.predict(X_input)[0]
    forecasts.append(pred)
    history.append(pred)

print(forecasts)