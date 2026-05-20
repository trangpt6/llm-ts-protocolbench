import pandas as pd
import numpy as np
import xgboost as xgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
train_df = df.iloc[:158]
test_df = df.iloc[158:]
def create_lag_features(data, lags=6):
    X = []
    y = []
    for i in range(lags, len(data)):
        row = []
        for lag in range(1, lags+1):
            row.append(data.iloc[i-lag, 1])
        for lag in range(1, lags+1):
            row.append(data.iloc[i-lag, 2])
        X.append(row)
        y.append(data.iloc[i, 2])
    return np.array(X), np.array(y)
X_train, y_train = create_lag_features(train_df, lags=6)
model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
history = train_df.copy()
forecasts = []
for i in range(len(test_df)):
    last_rows = history.iloc[-6:]
    X_pred = []
    for lag in range(1, 7):
        X_pred.append(last_rows.iloc[-lag, 1])
    for lag in range(1, 7):
        X_pred.append(last_rows.iloc[-lag, 2])
    X_pred = np.array(X_pred).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(float(pred))
    true_row = test_df.iloc[[i]]
    history = pd.concat([history, true_row], ignore_index=True)
    X_new, y_new = create_lag_features(history, lags=6)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_new, y_new)
print(forecasts)