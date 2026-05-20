import pandas as pd
import numpy as np
import lightgbm as lgb

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'

# Preprocessing: Parse Month and set as index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Define split based on Turn 0
total_timesteps = len(df)
train_size = 158
test_size = 40
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

def create_features(data, lags):
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data.iloc[i-lags:i].values.flatten())
        y.append(data.iloc[i])
    return np.array(X), np.array(y)

# Fixed Setup: Block-wise rolling update, Block size 12, Horizon 12, Retrain after each block
lags = 12
params = {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.05, 'importance_type': 'split', 'verbosity': -1}
all_forecasts = []

# Initial training data
current_train_series = df[target].iloc[:train_size]

# Forecasting loop
# Test set size is 40. Blocks of 12: 12, 12, 12, 4.
for start_idx in range(0, test_size, 12):
    # Prepare training data with lags
    X_train, y_train = create_features(current_train_series, lags)
    
    # Train LightGBM (Direct multi-step strategy usually implies separate models per horizon, 
    # but with block-wise rolling and ground truth update, we use a recursive-style 
    # application within the block or a single model for the block).
    # Given 'direct' strategy and block size 12, we train 12 models.
    models = []
    for h in range(1, 13):
        X_h, y_h = [], []
        for i in range(lags, len(current_train_series) - h + 1):
            X_h.append(current_train_series.iloc[i-lags:i].values)
            y_h.append(current_train_series.iloc[i+h-1])
        model = lgb.LGBMRegressor(**params)
        model.fit(np.array(X_h), np.array(y_h))
        models.append(model)
    
    # Forecast the next block (up to 12 steps)
    current_X = current_train_series.iloc[-lags:].values.reshape(1, -1)
    steps_to_forecast = min(12, test_size - start_idx)
    for h in range(steps_to_forecast):
        pred = models[h].predict(current_X)[0]
        all_forecasts.append(pred)
        
    # Update training set with ground truth from the block just forecasted
    actual_block = test_df[target].iloc[start_idx : start_idx + 12]
    current_train_series = pd.concat([current_train_series, actual_block])

print([round(f, 4) for f in all_forecasts])