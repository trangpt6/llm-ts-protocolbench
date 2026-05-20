import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Preprocessing: detect constant-value days (sensor freeze)
df['date_only'] = df['date'].dt.date
grouped = df.groupby('date_only')['OT']
frozen_days = grouped.transform('std') == 0
df.loc[frozen_days, 'OT'] = np.nan

# Replace negative values with NaN
df.loc[df['OT'] < 0, 'OT'] = np.nan

# Linear interpolation
df['OT'] = df['OT'].interpolate(method='linear')

# Remove helper column
df.drop('date_only', axis=1, inplace=True)

# Define train/test sizes
train_size = 14016
test_size = 3504
total = train_size + test_size

# Extract OT series
y = df['OT'].values

# Prepare lag features for training (one-step ahead)
lags = 24
X_list = []
y_list = []
for i in range(lags, train_size):
    X_list.append(y[i - lags:i])
    y_list.append(y[i])
X_train = np.array(X_list)
y_train = np.array(y_list)

# Train LightGBM model
model = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
model.fit(X_train, y_train)

# Recursive forecasting on test set
# Initial lag window: last lags values of training set
last_values = list(y[train_size - lags:train_size])
forecasts = []
for i in range(test_size):
    # Predict next step
    x_input = np.array(last_values[-lags:]).reshape(1, -1)
    pred = model.predict(x_input)[0]
    forecasts.append(pred)
    # Update lag window with prediction (no ground truth)
    last_values.append(pred)

print(forecasts)