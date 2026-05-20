import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']
lags = 6
def create_lag_features(data, lags, feature_cols, target):
    X = []
    y = []
    for i in range(lags, len(data)):
        lag_features = []
        for col in feature_cols:
            for lag in range(1, lags + 1):
                lag_features.append(data[col].iloc[i - lag])
        X.append(lag_features)
        y.append(data[target].iloc[i])
    return np.array(X), np.array(y)
train_size = 158
train_df = df.iloc[:train_size].reset_index(drop=True)
X_train, y_train = create_lag_features(train_df, lags, feature_cols, target)
model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
model.fit(X_train, y_train)
forecasts = []
current_data = train_df.copy()
for i in range(40):
    lag_features = []
    for col in feature_cols:
        for lag in range(1, lags + 1):
            lag_features.append(current_data[col].iloc[-lag])
    X_pred = np.array([lag_features])
    pred = model.predict(X_pred)[0]
    forecasts.append(float(pred))
    next_idx = train_size + i
    new_row = df.iloc[[next_idx]].reset_index(drop=True)
    current_data = pd.concat([current_data, new_row], ignore_index=True)
    X_train, y_train = create_lag_features(current_data, lags, feature_cols, target)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
print(forecasts)