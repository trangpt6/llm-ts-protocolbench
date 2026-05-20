import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])

# minimal preprocessing: convert empty strings in specified age columns to NaN
df['AGE 25-49'] = pd.to_numeric(df['AGE 25-49'], errors='coerce')
df['AGE 50-64'] = pd.to_numeric(df['AGE 50-64'], errors='coerce')

# ensure chronological order
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'
y = df[target_col].to_numpy()

# chronological 80/20 split
n_total = len(y)
train_size = int(0.8 * n_total)
y_train = y[:train_size]

# feature engineering: create lag features from target only
lags = 52
X_train = []
y_train_model = []
for i in range(lags, train_size):
    X_train.append(y_train[i - lags:i][::-1])
    y_train_model.append(y_train[i])

X_train = np.array(X_train)
y_train_model = np.array(y_train_model)

# train XGBoost with fixed hyperparameters
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train_model)

# recursive one-step ahead forecasting over the full test set
history = list(y_train)
forecasts = []
for _ in range(n_total - train_size):
    recent = history[-lags:]
    x_input = np.array(recent[::-1]).reshape(1, -1)
    pred = float(model.predict(x_input)[0])
    forecasts.append(pred)
    history.append(pred)

print(forecasts)