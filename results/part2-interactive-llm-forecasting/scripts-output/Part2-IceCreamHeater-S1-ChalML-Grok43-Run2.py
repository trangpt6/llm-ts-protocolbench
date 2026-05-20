import pandas as pd
import numpy as np
import lightgbm as lgb
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
target_col = 'Ice cream'
train_size = 158
train_series = df[target_col].iloc[:train_size]
lags = 12
train_data = pd.DataFrame({target_col: train_series})
for i in range(1, lags + 1):
    train_data[f'lag_{i}'] = train_data[target_col].shift(i)
train_data = train_data.dropna()
feature_cols = [f'lag_{i}' for i in range(1, lags + 1)]
X_train = train_data[feature_cols]
y_train = train_data[target_col]
model = lgb.LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)
current_lags = list(train_series.iloc[-12:].values)
forecasts = []
for _ in range(40):
    X_pred = np.array(current_lags).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    current_lags = current_lags[1:] + [pred]
print(forecasts)