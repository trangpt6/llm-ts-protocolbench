import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestRegressor
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
target = df['Passengers'].values
train_size = 115
train = target[:train_size]
test = target[train_size:]
forecasts = []
current_series = list(train)
for i in range(len(test)):
    X = []
    y = []
    for j in range(12, len(current_series)):
        X.append(current_series[j-12:j])
        y.append(current_series[j])
    X = np.array(X)
    y = np.array(y)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X, y)
    last_window = current_series[-12:]
    pred = []
    for h in range(12):
        X_pred = np.array(last_window).reshape(1, -1)
        next_val = model.predict(X_pred)[0]
        pred.append(next_val)
        last_window = last_window[1:] + [next_val]
    forecasts.append(pred[0])
    current_series.append(test[i])
print(forecasts)