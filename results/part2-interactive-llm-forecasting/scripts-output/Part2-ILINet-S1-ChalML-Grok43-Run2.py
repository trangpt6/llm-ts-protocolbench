import pandas as pd
import numpy as np
import xgboost as xgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target_col = '% WEIGHTED ILI'
cols_to_fill = [c for c in df.columns if c not in ['DATE', target_col]]
df[cols_to_fill] = df[cols_to_fill].fillna(method='ffill')
y = df[target_col].values.astype(float)
train_size = 1019
train_y = y[:train_size]
lags = 52
X_train = []
y_train = []
for i in range(lags, len(train_y)):
    X_train.append(train_y[i-lags:i])
    y_train.append(train_y[i])
X_train = np.array(X_train)
y_train = np.array(y_train)
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
current_lags = train_y[-lags:].tolist()
forecasts = []
for _ in range(255):
    X_pred = np.array([current_lags])
    pred = model.predict(X_pred)[0]
    forecasts.append(float(pred))
    current_lags = current_lags[1:] + [pred]
print(forecasts)