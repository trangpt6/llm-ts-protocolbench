import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
train_size = 158
test_size = 40
lags = 6
feature_cols = ['Heater', 'Ice cream']
target_col = 'Ice cream'
def create_lagged_features(data, lags, feature_cols, target_col):
    X = []
    y = []
    for i in range(lags, len(data)):
        lag_features = []
        for col in feature_cols:
            lag_features.extend(data[col].iloc[i-lags:i].values.tolist())
        X.append(lag_features)
        y.append(data[target_col].iloc[i])
    return np.array(X), np.array(y)
forecasts = []
current_data = df.iloc[:train_size].copy()
for step in range(test_size):
    X_train, y_train = create_lagged_features(current_data, lags, feature_cols, target_col)
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    last_lags = []
    for col in feature_cols:
        last_lags.extend(current_data[col].iloc[-lags:].values.tolist())
    next_pred = model.predict([last_lags])[0]
    forecasts.append(next_pred)
    next_row = df.iloc[[train_size + step]].copy()
    current_data = pd.concat([current_data, next_row], ignore_index=True)
print(forecasts)