import pandas as pd
import numpy as np
import random
import xgboost as xgb

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], index_col='Date')
target_col = 'Daily minimum temperatures'
df[target_col] = df[target_col].astype(float)

# Chronological split fixed by Turn 0
train_size = 2921
test_size = 731
train = df.iloc[:train_size]
# test = df.iloc[train_size:]  # not needed for training, but used for period

# Create lag features
lags = 14
X_list = []
y_list = []
for i in range(lags, len(train)):
    X_list.append(train[target_col].iloc[i-lags:i].values)
    y_list.append(train[target_col].iloc[i])

X_train = np.array(X_list)
y_train = np.array(y_list)

# Train XGBoost as per fixed hyperparameters
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# Recursive one-step ahead forecasting for the test set
last_lags = train[target_col].iloc[-lags:].values.tolist()
forecasts = []
for step in range(test_size):
    feats = np.array(last_lags[-lags:]).reshape(1, -1)
    pred = float(model.predict(feats)[0])
    forecasts.append(pred)
    last_lags.append(pred)

print(forecasts)