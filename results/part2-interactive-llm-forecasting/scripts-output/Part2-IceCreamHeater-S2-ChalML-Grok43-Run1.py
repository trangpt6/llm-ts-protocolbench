import pandas as pd
import numpy as np
import xgboost as xgb
import random
# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']
lags = 6
train_size = 158
current_train = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].reset_index(drop=True)
forecasts = []
for i in range(len(test_df)):
    X = []
    y = []
    for j in range(lags, len(current_train)):
        lag_vals = []
        for col in feature_cols:
            for lag in range(1, lags + 1):
                lag_vals.append(current_train[col].iloc[j - lag])
        X.append(lag_vals)
        y.append(current_train[target].iloc[j])
    X = np.array(X)
    y = np.array(y)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    last_lags = []
    for col in feature_cols:
        for lag in range(1, lags + 1):
            last_lags.append(current_train[col].iloc[-lag])
    X_pred = np.array([last_lags])
    pred = model.predict(X_pred)[0]
    forecasts.append(float(pred))
    new_row = test_df.iloc[i:i+1].copy()
    current_train = pd.concat([current_train, new_row], ignore_index=True)
print(forecasts)