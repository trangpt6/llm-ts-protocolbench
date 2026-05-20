import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
y = df['Daily minimum temperatures'].values

# Chronological split: first 2920 train, remaining 730 test
train_size = 2920
test_size = 730
y_train = y[:train_size]
y_test = y[train_size:]

# Rolling one-step ahead forecasting with retraining after each step
history = y_train.tolist()
forecasts = []
lags = 7

for i in range(test_size):
    # Build lag features from history
    X = np.array([history[j:j+lags] for j in range(len(history)-lags)])
    Y = np.array(history[lags:])
    
    # Fit LightGBM with fixed hyperparameters
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X, Y)
    
    # One-step ahead prediction
    x_pred = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(x_pred)[0]
    forecasts.append(float(pred))
    
    # Update history with observed ground truth
    history.append(float(y_test[i]))

print(forecasts)