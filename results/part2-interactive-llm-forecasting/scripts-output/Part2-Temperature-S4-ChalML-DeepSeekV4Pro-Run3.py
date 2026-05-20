import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)

# Read the CSV file (assumed to be in the current working directory as input.csv)
df = pd.read_csv(r'../../../data/Temperature.csv')

# Parse dates and set as index
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# Primary target
target = 'Daily minimum temperatures'
y = df[target].values

# Chronological 80/20 split
train_size = int(0.8 * len(y))
train_y = y[:train_size]
test_y = y[train_size:]

# Fixed hyperparameters
lags = 14
n_estimators = 200
max_depth = 5
learning_rate = 0.05
block_size = 30

# Function to create supervised learning set from a 1D time series
def create_supervised(series, lags):
    X, y_list = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y_list.append(series[i])
    return np.array(X), np.array(y_list)

# Initial training series (all available training data)
series = train_y.copy()
forecasts = []

# Proceed block by block over the test set
for start in range(0, len(test_y), block_size):
    end = min(start + block_size, len(test_y))
    actual_block = test_y[start:end]
    block_len = end - start

    # Prepare training data from the current series (which has all past true values)
    X_train, y_train = create_supervised(series, lags)

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Recursive multi-step forecasting for this block
    recent = list(series[-lags:])  # last lags values as initial context
    block_forecasts = []
    for _ in range(block_len):
        # Feature vector: [lag_1 (most recent), lag_2, ..., lag_14]
        X_pred = np.array([[recent[-i] for i in range(1, lags+1)]])
        pred = model.predict(X_pred)[0]
        block_forecasts.append(pred)
        recent.append(pred)
        recent = recent[-lags:]
    forecasts.extend(block_forecasts)

    # Append true values of this block to the series for future model updates
    series = np.append(series, actual_block)

# Output the forecast list
print(forecasts)