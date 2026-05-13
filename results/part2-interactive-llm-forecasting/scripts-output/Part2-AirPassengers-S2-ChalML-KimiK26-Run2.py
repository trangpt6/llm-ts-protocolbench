import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset and extract target
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].to_numpy()
n = len(y)
train_size = 115
history = list(y[:train_size])
forecasts = []

# rolling one-step ahead with retraining after each true observation
for i in range(train_size, n):
    train_series = np.array(history)
    X_train = []
    y_train = []
    for t in range(12, len(train_series)):
        X_train.append(train_series[t - 12:t][::-1])
        y_train.append(train_series[t])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    X_test = train_series[-12:][::-1].reshape(1, -1)
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    history.append(float(y[i]))

print(forecasts)