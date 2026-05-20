import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
passengers = df['Passengers'].tolist()
train_size = 115
train_list = passengers[:train_size]
test_list = passengers[train_size:]
forecasts = []
for i in range(len(test_list)):
    X_train = []
    y_train = []
    for j in range(12, len(train_list)):
        X_train.append(train_list[j-12:j])
        y_train.append(train_list[j])
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    last_lags = train_list[-12:]
    pred = model.predict([last_lags])[0]
    forecasts.append(pred)
    train_list.append(test_list[i])
print(forecasts)