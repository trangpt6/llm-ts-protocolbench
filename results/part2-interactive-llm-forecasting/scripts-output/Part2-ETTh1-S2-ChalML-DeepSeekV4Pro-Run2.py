import pandas as pd
import numpy as np
import xgboost as xgb
import random

random.seed(42)
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# Preprocessing: replace constant end-of-month days with NaN
bad_dates = ['2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
             '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
             '2017-08-31', '2017-10-31', '2017-12-31',
             '2018-01-31', '2018-03-31', '2018-05-31']
for d in bad_dates:
    if d in df.index:
        df.loc[d, 'OT'] = np.nan

# Chronological split
n = len(df)
train_size = int(0.8 * n)
train_idx = df.index[:train_size]
test_idx = df.index[train_size:]
train_series = df.loc[train_idx, 'OT'].copy()
test_series = df.loc[test_idx, 'OT'].copy()

known = train_series.copy()

def create_lagged_features(series, n_lags=12):
    df = pd.DataFrame({'y': series})
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    df = df.dropna(subset=['y'])
    return df

params = {
    'n_estimators': 50,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': 42,
    'verbosity': 0
}

forecasts = []

for t in range(len(test_series)):
    current_test_index = test_idx[t]
    train_df = create_lagged_features(known, n_lags=12)
    
    if len(train_df) == 0:
        forecast = 0.0
    else:
        X_train = train_df.drop(columns=['y'])
        y_train = train_df['y']
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        last_12 = known.iloc[-12:].values
        if len(last_12) < 12:
            last_12 = np.pad(last_12, (12 - len(last_12), 0), constant_values=np.nan)
        X_pred = pd.DataFrame([last_12], columns=[f'lag_{i}' for i in range(1, 13)])
        forecast = model.predict(X_pred)[0]
    
    forecasts.append(forecast)
    known.loc[current_test_index] = test_series.iloc[t]

print(forecasts)