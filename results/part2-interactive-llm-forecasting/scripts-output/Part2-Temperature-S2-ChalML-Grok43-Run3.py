import pandas as pd
import numpy as np
import lightgbm as lgb
# Set random seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
target = 'Daily minimum temperatures'
train_size = 2920
train_df = df.iloc[:train_size]
non_zero = train_df[train_df[target] > 0][target]
median_val = non_zero.median() if len(non_zero) > 0 else 0
df[target] = df[target].replace(0, median_val)
forecasts = []
current_train = df.iloc[:train_size][[target]].copy().reset_index(drop=True)
lags = 7
for i in range(730):
    for lag in range(1, lags + 1):
        current_train[f'lag_{lag}'] = current_train[target].shift(lag)
    train_feat = current_train.dropna()
    X_train = train_feat[[f'lag_{lag}' for lag in range(1, lags + 1)]]
    y_train = train_feat[target]
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    last_lags = current_train[target].iloc[-lags:].values[::-1]
    X_pred = pd.DataFrame([last_lags], columns=[f'lag_{lag}' for lag in range(1, lags + 1)])
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    true_val = df.iloc[train_size + i][target]
    current_train = pd.concat([current_train[[target]], pd.DataFrame({target: [true_val]})], ignore_index=True)
print(forecasts)