import pandas as pd
import numpy as np
import lightgbm as lgb

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'
data = df[target].values

# Define split based on Turn 0
total_timesteps = len(data)
train_size = 158
test_size = 40
train_data = data[:train_size]
test_data = data[train_size:]

# Fixed Hyperparameters from Turn 2
lags = 12
params = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'verbosity': -1,
    'random_state': 42
}
block_size = 12
horizon = 12

def create_features(series, n_lags):
    X, y = [], []
    for i in range(len(series) - n_lags):
        X.append(series[i:i + n_lags])
        y.append(series[i + n_lags])
    return np.array(X), np.array(y)

def train_direct_models(current_train_series, n_lags, h_max, model_params):
    models = []
    for h in range(1, h_max + 1):
        X, y = [], []
        for i in range(len(current_train_series) - n_lags - h + 1):
            X.append(current_train_series[i:i + n_lags])
            y.append(current_train_series[i + n_lags + h - 1])
        model = lgb.LGBMRegressor(**model_params)
        model.fit(np.array(X), np.array(y))
        models.append(model)
    return models

all_forecasts = []
current_history = list(train_data)

# Block-wise rolling update with retraining
for start_idx in range(0, test_size, block_size):
    # Retrain models on all available data before each block
    models = train_direct_models(np.array(current_history), lags, horizon, params)
    
    # Forecast the next block
    last_window = np.array(current_history[-lags:]).reshape(1, -1)
    block_preds = []
    for h in range(horizon):
        pred = models[h].predict(last_window)[0]
        block_preds.append(pred)
    
    # Determine how many steps to take from this block (handle end of test set)
    steps_to_take = min(block_size, test_size - start_idx)
    actual_preds = block_preds[:steps_to_take]
    all_forecasts.extend(actual_preds)
    
    # Update history with ground truth for the next block's retraining
    actual_observations = test_data[start_idx:start_idx + steps_to_take]
    current_history.extend(actual_observations)

print(all_forecasts)