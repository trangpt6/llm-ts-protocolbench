import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load and prepare the data
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)
df.set_index('DATE', inplace=True)
target_col = '% WEIGHTED ILI'
series = df[target_col].astype(float)

# Chronological split: 80% train, 20% test
train_size = int(0.8 * len(series))
train = series.iloc[:train_size].values
test = series.iloc[train_size:].values

# Initialize training series with all training data
training_series = list(train)

# One-step-ahead rolling forecasting with retraining
forecasts = []
for i in range(len(test)):
    # Build supervised samples from current training series
    L = len(training_series)
    X_train = []
    y_train = []
    for j in range(13, L):
        X_train.append(training_series[j-13:j])
        y_train.append(training_series[j])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train LightGBM with fixed hyperparameters
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predict next value using last 13 lags
    X_pred = np.array([training_series[-13:]])
    y_pred = model.predict(X_pred)[0]
    forecasts.append(y_pred)
    
    # Update training set with true test value (ground truth)
    training_series.append(test[i])

print(forecasts)