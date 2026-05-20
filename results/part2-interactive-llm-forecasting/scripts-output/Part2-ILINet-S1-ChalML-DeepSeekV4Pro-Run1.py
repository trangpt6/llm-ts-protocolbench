import pandas as pd
import numpy as np
import xgboost as xgb

# Seed for reproducibility
np.random.seed(42)

# Load and preprocess
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'
series = df[target_col].astype(float).values

# Train/test split (chronological, 80/20)
train_size = 1050
train = series[:train_size]
test = series[train_size:]

# Create lagged dataset from training data
lags = 52
X_train = []
y_train = []
for i in range(lags, len(train)):
    X_train.append(train[i-lags:i])
    y_train.append(train[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Model with fixed hyperparameters
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# Recursive one-step ahead forecasting over test set
history = list(train[-lags:])  # last lags values from training
forecasts = []
for _ in range(len(test)):
    # Prepare feature vector from the most recent lags
    feats = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(feats)[0]
    forecasts.append(pred)
    # Update history with prediction (no ground truth)
    history.append(pred)

# Output the forecasts as a list
print([float(x) for x in forecasts])