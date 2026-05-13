import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values.astype(float)
n = len(y)   # 144
train_size = int(0.8 * n)  # 115

forecasts = []
for step in range(n - train_size):
    current_idx = train_size + step        # first step: 115 (index of first test point)
    series = y[:current_idx]               # all data up to (but not including) the point being forecast
    # Build training set from lag 12 to current_idx-1
    X_train = []
    y_train = []
    for i in range(12, len(series)):
        X_train.append(series[i-12:i])
        y_train.append(series[i])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                         objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)
    
    # Use last 12 values as predictor
    last_12 = series[-12:]
    X_pred = np.array([last_12])
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)

print(forecasts)