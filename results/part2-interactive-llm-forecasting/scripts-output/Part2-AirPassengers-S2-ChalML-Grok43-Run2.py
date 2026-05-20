import pandas as pd
import numpy as np
import xgboost as xgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values
train_size = 115
train = data[:train_size]
test = data[train_size:]
forecasts = []
current_train = list(train)
for _ in range(len(test)):
    X_train = []
    y_train = []
    for j in range(12, len(current_train)):
        X_train.append(current_train[j-12:j])
        y_train.append(current_train[j])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    model.fit(X_train, y_train)
    last_12 = np.array(current_train[-12:]).reshape(1, -1)
    pred = model.predict(last_12)[0]
    forecasts.append(float(pred))
    current_train.append(test[_])
print(forecasts)