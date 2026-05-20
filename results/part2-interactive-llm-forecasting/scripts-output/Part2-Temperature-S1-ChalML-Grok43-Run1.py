import pandas as pd
import numpy as np
import xgboost as xgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target_col = 'Daily minimum temperatures'
data = df[target_col].values
train_size = 2920
train_data = data[:train_size]
lags = 14
X_train = []
y_train = []
for i in range(lags, len(train_data)):
    X_train.append(train_data[i-lags:i])
    y_train.append(train_data[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
forecasts = []
current_window = list(train_data[-lags:])
for _ in range(730):
    pred = model.predict(np.array([current_window]))[0]
    forecasts.append(float(pred))
    current_window = current_window[1:] + [pred]
print(forecasts)