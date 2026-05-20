import pandas as pd
import numpy as np
import random
from xgboost import XGBRegressor

np.random.seed(42)
random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()
target = 'Ice cream'
exog = 'Heater'

train_size = int(0.8 * len(df))
test_start_idx = train_size
test_indices = df.index[test_start_idx:]

def create_features(data, lags=6):
    feat = pd.DataFrame(index=data.index)
    for col in [exog, target]:
        for lag in range(1, lags + 1):
            feat[f'{col}_lag{lag}'] = data[col].shift(lag)
    return feat

forecasts = []

for test_idx in test_indices:
    train_end_loc = df.index.get_loc(test_idx) - 1
    train_data = df.iloc[:train_end_loc + 1]

    full_feat = create_features(train_data, lags=6)
    target_series = train_data[target]
    mask = full_feat.notna().all(axis=1) & target_series.notna()
    train_X = full_feat[mask]
    train_y = target_series[mask]

    last_six = train_data.iloc[-6:]
    test_feat = {}
    for col in [exog, target]:
        for lag in range(1, 7):
            test_feat[f'{col}_lag{lag}'] = last_six[col].iloc[-lag]
    test_X = pd.DataFrame([test_feat])

    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)[0]
    forecasts.append(pred)

print(forecasts)