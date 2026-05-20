import pandas as pd
import numpy as np
import lightgbm as lgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'
exog = 'Heater'
train_size = 158
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]
def create_features(data, lags=12):
    X = []
    y = []
    for i in range(lags, len(data)):
        ice_lags = data[target].iloc[i-lags:i].tolist()
        heat_lags = data[exog].iloc[i-lags:i].tolist()
        X.append(ice_lags + heat_lags)
        y.append(data[target].iloc[i])
    return np.array(X), np.array(y)
X_train, y_train = create_features(train_df, 12)
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
forecasts = []
ice_lags = train_df[target].iloc[-12:].tolist()
heat_lags = train_df[exog].iloc[-12:].tolist()
for i in range(40):
    features = np.array(ice_lags + heat_lags).reshape(1, -1)
    pred = model.predict(features)[0]
    forecasts.append(pred)
    ice_lags = ice_lags[1:] + [pred]
    next_heat = test_df[exog].iloc[i]
    heat_lags = heat_lags[1:] + [next_heat]
print(forecasts)