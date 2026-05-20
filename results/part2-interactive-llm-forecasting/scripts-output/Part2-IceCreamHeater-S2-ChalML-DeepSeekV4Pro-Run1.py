import random
import numpy as np
random.seed(42)
np.random.seed(42)

import pandas as pd
from xgboost import XGBRegressor

# Read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])

# Chronological split based on fixed train size 158
train_df = df.iloc[:158].copy()
test_df = df.iloc[158:].copy()

# Hyperparameters
lags = 6
n_estimators = 50
max_depth = 3
learning_rate = 0.1

# Lists to store forecasts
forecasts = []

# Initial dataset that grows with true values after each step
rolling_data = train_df.copy()

for i in range(len(test_df)):
    # Current observation to be predicted
    current_test = test_df.iloc[i]
    # Build features from the latest available data (rolling_data)
    # Use both Heater and Ice cream columns
    series = rolling_data[['Heater', 'Ice cream']].values
    # We need at least lags rows to create features
    if len(series) < lags + 1:
        raise ValueError("Not enough data to create lag features")
    # Create features for all possible instances in rolling_data
    X_list = []
    y_list = []
    for j in range(lags, len(series)):
        # Extract lag features for Heater and Ice cream: shape (lags*2,)
        feat = series[j-lags:j, 0:].flatten()  # Heater lags then Ice cream lags
        X_list.append(feat)
        y_list.append(series[j, 1])  # target is Ice cream at current time
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Train model
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, random_state=42)
    model.fit(X, y)
    
    # Build feature for prediction: latest lags from the last available rows
    last_lags = series[-lags:, 0:].flatten().reshape(1, -1)
    pred = model.predict(last_lags)[0]
    forecasts.append(pred)
    
    # Append the true observation to rolling data for next step
    rolling_data = pd.concat([rolling_data, current_test.to_frame().T], ignore_index=True)

# Output the forecast list
print(forecasts)