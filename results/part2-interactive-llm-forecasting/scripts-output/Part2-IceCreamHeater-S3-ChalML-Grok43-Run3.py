import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'
feature_cols = ['Heater', target]
lags = 6
n_estimators = 50
max_depth = 4
train_size = 158
history = df.iloc[:train_size].copy()
forecasts = []
for i in range(40):
    X_list = []
    y_list = []
    for j in range(lags, len(history)):
        row = []
        for col in feature_cols:
            for lag in range(1, lags + 1):
                row.append(history[col].iloc[j - lag])
        X_list.append(row)
        y_list.append(history[target].iloc[j])
    X_train = np.array(X_list)
    y_train = np.array(y_list)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    row = []
    for col in feature_cols:
        for lag in range(1, lags + 1):
            row.append(history[col].iloc[-lag])
    X_pred = np.array(row).reshape(1, -1)
    pred = model.predict(X_pred)[0]
    forecasts.append(pred)
    next_row = df.iloc[train_size + i].copy()
    history = pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)
print(forecasts)