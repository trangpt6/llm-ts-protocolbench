import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds
random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
y = df['Daily minimum temperatures'].values

# Split
train_size = int(0.8 * len(y))
train_y = y[:train_size]
test_y = y[train_size:]

lags = 7
horizon = 7

forecasts = []

# Rolling update with retraining
for i in range(len(test_y)):
    current_y = y[:train_size + i]
    
    step_forecasts = []
    for h in range(1, horizon + 1):
        X_train = []
        Y_train = []
        for j in range(lags, len(current_y) - h + 1):
            X_train.append(current_y[j-lags:j])
            Y_train.append(current_y[j+h-1])
            
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        model.fit(X_train, Y_train)
        
        X_test = current_y[-lags:].reshape(1, -1)
        pred = model.predict(X_test)[0]
        step_forecasts.append(float(pred))
        
    forecasts.extend(step_forecasts)

print(forecasts)