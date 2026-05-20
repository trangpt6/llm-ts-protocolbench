import pandas as pd
import numpy as np
import lightgbm as lgb
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = 'Daily minimum temperatures'
for idx in df[df[target] == 0].index:
    if idx >= 7:
        df.loc[idx, target] = df.loc[idx-7:idx-1, target].median()
train_size = 2920
train_df = df.iloc[:train_size][[target]].reset_index(drop=True)
test_df = df.iloc[train_size:][[target]].reset_index(drop=True)
forecasts = []
current_data = train_df.copy()
for step in range(len(test_df)):
    data_with_lags = current_data.copy()
    for lag in range(1, 8):
        data_with_lags[f'lag{lag}'] = data_with_lags[target].shift(lag)
    data_with_lags = data_with_lags.dropna()
    X_train = data_with_lags[[f'lag{lag}' for lag in range(1, 8)]]
    y_train = data_with_lags[target]
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    last_vals = current_data[target].tail(7).values.tolist()
    features = last_vals[::-1]
    pred = model.predict([features])[0]
    forecasts.append(pred)
    true_val = test_df.loc[step, target]
    new_row = pd.DataFrame({target: [true_val]})
    current_data = pd.concat([current_data, new_row], ignore_index=True)
print(forecasts)