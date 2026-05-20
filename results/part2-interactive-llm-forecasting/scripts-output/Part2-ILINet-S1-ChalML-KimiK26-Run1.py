import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/ILINet.csv', na_values=[''])

# preprocessing
df.columns = df.columns.str.strip()
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
target_col = '% WEIGHTED ILI'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

# chronological split
train_size = 1043
test_size = 261
train = df.iloc[:train_size].copy()

y = train[target_col].values

# build lag features for training
lags = 52
X_train = []
y_train = []
for i in range(lags, len(train)):
    X_train.append(y[i - lags:i][::-1])
    y_train.append(y[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# train xgboost with fixed hyperparameters
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# recursive one-step ahead forecasting over the test set
history = y[-lags:].tolist()
forecasts = []
for i in range(test_size):
    X_i = np.array(history[-lags:][::-1]).reshape(1, -1)
    pred = model.predict(X_i)[0]
    forecasts.append(float(pred))
    history.append(pred)

print(forecasts)