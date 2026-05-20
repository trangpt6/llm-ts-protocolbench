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
train_data = data[:train_size].tolist()
test_data = data[train_size:]
forecasts = []
for step in range(len(test_data)):
    lags = 12
    X = []
    y = []
    for i in range(lags, len(train_data)):
        X.append(train_data[i-lags:i])
        y.append(train_data[i])
    X = np.array(X)
    y = np.array(y)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    last_window = np.array(train_data[-lags:]).reshape(1, -1)
    pred = model.predict(last_window)[0]
    forecasts.append(float(pred))
    train_data.append(test_data[step])
print(forecasts)