import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

# Reproducibility seeds
random.seed(42)
np.random.seed(42)

# Load and prepare data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.sort_values('Month', inplace=True)
series = df['Passengers'].values

# Chronological split
train_len = int(0.8 * len(series))
# train_len = 115, test start index 115

forecasts = []

for t in range(train_len, len(series)):
    # train on all data up to index t-1
    y_train = series[:t]
    n = len(y_train)
    # create lag features for training (12 lags -> predict next)
    X_train = np.array([y_train[i-12:i] for i in range(12, n)])
    y_train_lagged = y_train[12:]
    # Train model
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train_lagged)
    # Recursive forecast of horizon 12
    current_lags = list(y_train[n-12:n])
    forecast_h = []
    for _ in range(12):
        pred = model.predict([current_lags])[0]
        forecast_h.append(pred)
        # slide window: drop oldest, add new prediction
        current_lags = current_lags[1:] + [pred]
    # The first forecast is the one-step-ahead prediction for month t
    forecasts.append(forecast_h[0])

# Output the flat list
print(forecasts)