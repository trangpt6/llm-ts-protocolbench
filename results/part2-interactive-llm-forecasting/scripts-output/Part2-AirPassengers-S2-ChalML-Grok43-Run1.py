import pandas as pd
import numpy as np
import xgboost as xgb
import random
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
target = 'Passengers'
train_size = 115
train_data = df[target].iloc[:train_size].tolist()
test_data = df[target].iloc[train_size:].tolist()
forecasts = []
for step in range(len(test_data)):
    X_train = []
    y_train = []
    for i in range(12, len(train_data)):
        X_train.append(train_data[i-12:i])
        y_train.append(train_data[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    last_lags = np.array(train_data[-12:]).reshape(1, 12)
    pred = model.predict(last_lags)[0]
    forecasts.append(float(pred))
    train_data.append(test_data[step])
print(forecasts)