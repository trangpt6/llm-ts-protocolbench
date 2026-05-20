import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

target_col = 'Daily minimum temperatures'
y = df[target_col].values

# Fixed chronological split from Turn 0
train_size = 2920
test_size = 730
lags = 7

forecasts = []
train_series = y[:train_size].copy()

for i in range(train_size, train_size + test_size):
    X_train = []
    Y_train = []
    for j in range(lags, len(train_series)):
        X_train.append(train_series[j-lags:j])
        Y_train.append(train_series[j])
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, Y_train)
    
    x_pred = train_series[-lags:].reshape(1, -1)
    y_pred = model.predict(x_pred)[0]
    forecasts.append(float(y_pred))
    
    train_series = np.append(train_series, y[i])

print(forecasts)