import pandas as pd
import numpy as np
import xgboost as xgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']
lags = 6
train_size = 158
current_data = df.iloc[:train_size].copy()
forecasts = []
for i in range(40):
    X_train = []
    y_train = []
    for j in range(lags, len(current_data)):
        lag_features = []
        for col in feature_cols:
            for lag in range(1, lags + 1):
                lag_features.append(current_data[col].iloc[j - lag])
        X_train.append(lag_features)
        y_train.append(current_data[target_col].iloc[j])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    lag_features_pred = []
    for col in feature_cols:
        for lag in range(1, lags + 1):
            lag_features_pred.append(current_data[col].iloc[-lag])
    X_pred = np.array([lag_features_pred])
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    new_row = df.iloc[train_size + i].copy()
    current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
print(forecasts)