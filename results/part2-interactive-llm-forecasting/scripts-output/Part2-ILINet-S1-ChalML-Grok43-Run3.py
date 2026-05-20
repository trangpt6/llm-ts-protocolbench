import pandas as pd
import numpy as np
import xgboost as xgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
non_target_cols = [col for col in df.columns if col != '% WEIGHTED ILI']
df[non_target_cols] = df[non_target_cols].fillna(method='ffill')
target = '% WEIGHTED ILI'
train_size = 1019
train = df[target].iloc[:train_size].values
def create_lags(series, lags=52):
    X = []
    y = []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)
X_train, y_train = create_lags(train, 52)
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
forecasts = []
current_lags = train[-52:].tolist()
for _ in range(255):
    pred = model.predict(np.array([current_lags]))[0]
    forecasts.append(pred)
    current_lags = current_lags[1:] + [pred]
print(forecasts)