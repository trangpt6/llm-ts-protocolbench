import pandas as pd
import numpy as np
import lightgbm as lgb
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds
random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
y = df['Daily minimum temperatures'].values

# Train test split
train_size = int(len(y) * 0.8)
train_y = y[:train_size]
test_y = y[train_size:]

# Hyperparameters
lags = 7
H = 7
n_estimators = 50
max_depth = 3
learning_rate = 0.1

history = list(train_y)
test_y_list = list(test_y)
forecasts = []

# Rolling update and retraining
for t in range(len(test_y_list)):
    current_forecast = []
    for h in range(1, H + 1):
        X_train, Y_train = [], []
        for i in range(len(history) - lags - h + 1):
            X_train.append(history[i : i + lags])
            Y_train.append(history[i + lags + h - 1])
        
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, Y_train)
        
        X_test = np.array([history[-lags:]])
        pred = float(model.predict(X_test)[0])
        current_forecast.append(pred)
    
    forecasts.extend(current_forecast)
    history.append(test_y_list[t])

# Print final forecasts
print(forecasts)