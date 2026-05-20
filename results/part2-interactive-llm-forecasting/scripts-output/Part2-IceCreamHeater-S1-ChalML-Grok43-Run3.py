import pandas as pd
import numpy as np
import lightgbm as lgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
train_size = 158
train = df[target_col].iloc[:train_size].values
def create_lag_features(series, lags=12):
    X = []
    y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)
X_train, y_train = create_lag_features(train, lags=12)
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
forecasts = []
current_window = list(train[-12:])
for _ in range(40):
    X_pred = np.array(current_window).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    current_window.pop(0)
    current_window.append(pred)
print(forecasts)