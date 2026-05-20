import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.tolist()
train_size = 115
current_series = data[:train_size]
test_size = len(data) - train_size
forecasts = []
for i in range(test_size):
    lags = 12
    X = []
    y = []
    for j in range(lags, len(current_series)):
        X.append(current_series[j-lags:j])
        y.append(current_series[j])
    X = np.array(X)
    y = np.array(y)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)
    last_lags = np.array(current_series[-lags:]).reshape(1, -1)
    pred = model.predict(last_lags)[0]
    forecasts.append(pred)
    current_series.append(data[train_size + i])
print(forecasts)