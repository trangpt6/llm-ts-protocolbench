import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']
lags = 6

n = len(df)
train_size = int(0.8 * n)

def build_features(data, lags, feature_cols, target_col):
    X_list = []
    y_list = []
    for i in range(lags, len(data)):
        feats = []
        for col in feature_cols:
            for l in range(1, lags + 1):
                feats.append(data[col].iloc[i - l])
        X_list.append(feats)
        y_list.append(data[target_col].iloc[i])
    return np.array(X_list), np.array(y_list)

forecasts = []
for step in range(n - train_size):
    end_idx = train_size + step
    train_data = df.iloc[:end_idx]
    X_train, y_train = build_features(train_data, lags, feature_cols, target_col)

    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    feats = []
    for col in feature_cols:
        for l in range(1, lags + 1):
            feats.append(df[col].iloc[end_idx - l])
    X_test = np.array(feats).reshape(1, -1)
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))

print(forecasts)