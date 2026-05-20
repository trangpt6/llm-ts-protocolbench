import random
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

total = len(df)
train_size = int(0.8 * total)
train = df.iloc[:train_size].reset_index(drop=True)
test = df.iloc[train_size:].reset_index(drop=True)

target_col = 'Ice cream'
exog_col = 'Heater'
lags = 12

def build_features(series_a, series_b, lags):
    X, y = [], []
    for i in range(lags, len(series_a)):
        feats = list(series_a[i-lags:i]) + list(series_b[i-lags:i])
        X.append(feats)
        y.append(series_a[i])
    return np.array(X), np.array(y)

train_target = train[target_col].values.astype(float)
train_exog = train[exog_col].values.astype(float)

X_tr_t, y_tr_t = build_features(train_target, train_exog, lags)
X_tr_e, y_tr_e = build_features(train_exog, train_target, lags)

model_t = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
model_t.fit(X_tr_t, y_tr_t)

model_e = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
model_e.fit(X_tr_e, y_tr_e)

hist_t = list(train_target)
hist_e = list(train_exog)

forecasts = []
for _ in range(len(test)):
    feat_t = np.array(hist_t[-lags:] + hist_e[-lags:]).reshape(1, -1)
    feat_e = np.array(hist_e[-lags:] + hist_t[-lags:]).reshape(1, -1)
    p_t = float(model_t.predict(feat_t)[0])
    p_e = float(model_e.predict(feat_e)[0])
    forecasts.append(p_t)
    hist_t.append(p_t)
    hist_e.append(p_e)

print(forecasts)