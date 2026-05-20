import pandas as pd
import numpy as np
import lightgbm as lgb

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'

# Preprocessing: Parse Month and set as index
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Define split based on Turn 0
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

# Fixed Hyperparameters and Setup
lags = 12
params = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'importance_type': 'split',
    'random_state': 42,
    'n_jobs': 1
}
block_size = 12
horizon = 12

def create_features(data, target_col, n_lags):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[target_col].iloc[i-n_lags:i].values)
        y.append(data[target_col].iloc[i])
    return np.array(X), np.array(y)

all_forecasts = []
current_train_end = train_size

# Block-wise rolling update with retraining
while len(all_forecasts) < test_size:
    # Prepare training data up to current_train_end
    train_df = df.iloc[:current_train_end].copy()
    
    # Direct multi-step strategy: train a separate model for each step in the horizon
    block_preds = []
    for h in range(1, horizon + 1):
        # For each horizon step h, target is y(t+h), features are lags up to y(t)
        X_train, y_train = [], []
        data_vals = train_df[target].values
        for i in range(lags, len(data_vals) - h + 1):
            X_train.append(data_vals[i-lags:i])
            y_train.append(data_vals[i + h - 1])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predict for the test block
        # Features for the very first prediction of the block use the last 'lags' of train_df
        X_test = data_vals[-lags:].reshape(1, -1)
        pred = model.predict(X_test)[0]
        block_preds.append(pred)
    
    # Only take the number of predictions needed to cover the test set or the block
    remaining = test_size - len(all_forecasts)
    take = min(block_size, remaining)
    all_forecasts.extend(block_preds[:take])
    
    # Update training set with ground truth (block-wise)
    current_train_end += take

# Ensure output is exactly the test size
final_forecasts = [float(x) for x in all_forecasts[:test_size]]
print(final_forecasts)