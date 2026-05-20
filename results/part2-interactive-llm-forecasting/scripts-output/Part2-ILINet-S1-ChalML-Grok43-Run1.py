import pandas as pd
import numpy as np
import random
import xgboost as xgb
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target_col = '% WEIGHTED ILI'
non_target_cols = [col for col in df.columns if col != target_col and col != 'DATE']
df[non_target_cols] = df[non_target_cols].fillna(method='ffill')
train_size = 1019
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()
lags = 52
def create_lags(series, lags):
    df_lags = pd.DataFrame()
    for lag in range(1, lags+1):
        df_lags[f'lag_{lag}'] = series.shift(lag)
    return df_lags.dropna()
train_target = train_df[target_col]
train_features = create_lags(train_target, lags)
train_y = train_target.iloc[lags:]
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
model.fit(train_features, train_y)
forecasts = []
current_series = train_target.tolist()
for i in range(len(test_df)):
    last_lags = current_series[-lags:]
    feat = np.array(last_lags).reshape(1, -1)
    pred = model.predict(feat)[0]
    forecasts.append(pred)
    current_series.append(pred)
print(forecasts)