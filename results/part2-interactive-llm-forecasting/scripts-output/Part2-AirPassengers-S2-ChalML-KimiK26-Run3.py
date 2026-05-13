import random
import numpy as np
import pandas as pd
import xgboost as xgb

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].astype(float).values
train_size = 115
lags = 12

train_y = y[:train_size].copy()
forecasts = []

for i in range(train_size, len(y)):
    X_train = np.array([train_y[j - lags:j] for j in range(lags, len(train_y))])
    y_train = train_y[lags:]
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1
    )
    model.fit(X_train, y_train)
    x_pred = train_y[-lags:].reshape(1, -1)
    pred = model.predict(x_pred)[0]
    forecasts.append(float(pred))
    true_val = y[i]
    train_y = np.append(train_y, true_val)

print(forecasts)