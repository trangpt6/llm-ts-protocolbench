import numpy as np
import pandas as pd
import xgboost as xgb

np.random.seed(42)

# Load and prepare data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
y = df['Daily minimum temperatures'].values

# Train/test split (80 % train, 20 % test)
train_size = int(0.8 * len(y))
train_y = y[:train_size]
test_y = y[train_size:]

# Fixed hyperparameters
lags = 14
n_estimators = 300
max_depth = 6
learning_rate = 0.05

# Create lagged features for training
def create_lagged_features(series, lags):
    X = np.zeros((len(series) - lags, lags))
    for i in range(lags, len(series)):
        X[i - lags, :] = series[i - lags:i][::-1]  # most recent first
    y_lag = series[lags:]
    return X, y_lag

X_train, y_train = create_lagged_features(train_y, lags)

# Model training
model = xgb.XGBRegressor(n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         random_state=42)
model.fit(X_train, y_train)

# Recursive one-step-ahead forecasting over the full test set
forecasts = []
last_window = train_y[-lags:].tolist()  # chronological order, oldest first

for _ in range(len(test_y)):
    # Prepare feature vector: most recent first, matching training order
    X_input = np.array(last_window[::-1]).reshape(1, -1)
    pred = model.predict(X_input)[0]
    forecasts.append(pred)
    last_window.pop(0)  # remove oldest element
    last_window.append(pred)

print(forecasts)