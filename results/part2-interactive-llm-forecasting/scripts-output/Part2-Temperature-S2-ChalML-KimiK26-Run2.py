import random
random.seed(42)
import numpy as np
np.random.seed(42)
import pandas as pd
import lightgbm as lgb

# Read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
y = df['Daily minimum temperatures'].values

# Define split parameters
total = len(y)
train_size = int(0.8 * total)
test_size = total - train_size
lags = 7

# Initialize history with training portion
history = y[:train_size].tolist()
forecasts = []

for i in range(test_size):
    hist_arr = np.array(history)
    L = len(hist_arr)
    
    # Build supervised lag matrix
    X_train = np.zeros((L - lags, lags))
    for lag in range(1, lags + 1):
        X_train[:, lag - 1] = hist_arr[lags - lag:L - lag]
    y_train = hist_arr[lags:]
    
    # Fit LightGBM with fixed hyperparameters
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    # Form prediction input from most recent lags
    x_pred = np.zeros((1, lags))
    for lag in range(1, lags + 1):
        x_pred[0, lag - 1] = hist_arr[-lag]
    pred = model.predict(x_pred)[0]
    forecasts.append(float(pred))
    
    # Append true observation for rolling retraining
    history.append(y[train_size + i])

print(forecasts)