import numpy as np
import pandas as pd
import xgboost as xgb

# set random seeds for reproducibility
np.random.seed(42)

# load and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False)
df = df.sort_values('Date').reset_index(drop=True)
df.set_index('Date', inplace=True)
target = df['Daily minimum temperatures']

# split indices (chronological 80/20)
n = len(target)
train_size = int(0.8 * n)  # 2921
train_end_idx = train_size - 1  # index 2920 (date: 1988-12-30)
test_start_idx = train_size     # index 2921 (date: 1988-12-31)
test_size = n - train_size      # 731

# create lagged dataset for training: use 14 lags
lags = 14
X_train = []
y_train = []
for i in range(lags, train_size):
    X_train.append(target.iloc[i-lags:i].values)
    y_train.append(target.iloc[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# define model with fixed hyperparameters
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    objective='reg:squarederror',
    verbosity=0
)

# train on entire training set
model.fit(X_train, y_train)

# initial history for recursive forecasting: last 14 training values
history = list(target.iloc[train_size-lags:train_size].values)  # indices 2907..2920

# forecast recursively for test horizon
forecasts = []
for step in range(test_size):
    # use last lags from history
    X_input = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(X_input)[0]
    forecasts.append(pred)
    # append prediction to history (no ground truth)
    history.append(pred)

# output final forecasts as flat list
print(forecasts)