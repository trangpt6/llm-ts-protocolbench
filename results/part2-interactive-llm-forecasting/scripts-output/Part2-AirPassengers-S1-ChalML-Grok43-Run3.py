import pandas as pd
import numpy as np
import lightgbm as lgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
target = df['Passengers']
train_size = 115
train = target.iloc[:train_size]
def create_lag_features(series, lags=12):
    data = pd.DataFrame()
    for i in range(1, lags + 1):
        data[f'lag_{i}'] = series.shift(i)
    data['target'] = series
    return data.dropna()
train_df = create_lag_features(train, 12)
X_train = train_df[[f'lag_{i}' for i in range(1,13)]]
y_train = train_df['target']
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
last_12 = list(train.values[-12:])
forecasts = []
for _ in range(29):
    features = np.array(last_12[::-1]).reshape(1, 12)
    pred = model.predict(features)[0]
    forecasts.append(pred)
    last_12.pop(0)
    last_12.append(pred)
print(forecasts)