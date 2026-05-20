import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# preprocessing: minimal, retain all values as-is
# no missing values, no duplicated timestamps, keep anomalous spikes

# define target and features
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']

# train/test split: first 157 rows train, remaining 40 test
train_size = 157
test_size = 40
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# create lag features for direct multi-step strategy
def create_lag_features(data, target_col, feature_cols, lags):
    feat_df = data[feature_cols].copy()
    lagged = pd.DataFrame()
    for col in feature_cols:
        for lag in range(1, lags + 1):
            lagged[f'{col}_lag{lag}'] = feat_df[col].shift(lag)
    lagged[target_col] = data[target_col].values
    return lagged

# direct multi-step: train 12 models, one per horizon
# block-wise rolling with retraining after each block
# block size = 12, horizon = 12

lags = 12
block_size = 12
horizon = 12

# initial training data
current_train = train_df.copy()

forecasts = []

# number of complete blocks in test set
num_blocks = test_size // block_size  # 3 blocks (36 steps)
remainder = test_size % block_size  # 4 remaining steps

test_idx = 0

for block in range(num_blocks + 1):
    # prepare training features
    train_lagged = create_lag_features(current_train, target_col, feature_cols, lags)
    train_lagged = train_lagged.dropna()
    
    X_train = train_lagged.drop(columns=[target_col]).values
    y_train = train_lagged[target_col].values
    
    # for direct multi-step, we need to predict h steps ahead
    # build targets for each horizon using shifted target
    max_horizon = min(horizon, test_size - test_idx)
    if max_horizon <= 0:
        break
    
    # train one model per horizon step
    models = []
    for h in range(1, max_horizon + 1):
        # target is h steps ahead from current position
        y_h = train_lagged[target_col].shift(-h).dropna()
        valid_idx = y_h.index
        X_h = train_lagged.loc[valid_idx].drop(columns=[target_col]).values
        y_h_vals = y_h.values
        
        if len(y_h_vals) == 0:
            # fallback: use last known value
            models.append(None)
            continue
        
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_h, y_h_vals)
        models.append(model)
    
    # forecast for this block
    # start from last known values
    last_known = current_train[feature_cols].copy()
    
    block_forecasts = []
    for h in range(1, max_horizon + 1):
        # build feature vector from last lags
        feat_vec = []
        for col in feature_cols:
            for lag in range(1, lags + 1):
                idx = len(last_known) - lag
                if idx >= 0:
                    feat_vec.append(last_known[col].iloc[idx])
                else:
                    feat_vec.append(last_known[col].iloc[0])
        
        X_pred = np.array(feat_vec).reshape(1, -1)
        
        if models[h-1] is not None:
            pred = models[h-1].predict(X_pred)[0]
        else:
            pred = last_known[target_col].iloc[-1]
        
        block_forecasts.append(pred)
        
        # update last_known with this prediction for next step's lags
        new_row = pd.DataFrame({col: [pred if col == target_col else last_known[col].iloc[-1]] for col in feature_cols})
        last_known = pd.concat([last_known, new_row], ignore_index=True)
    
    forecasts.extend(block_forecasts)
    
    # update with true values and retrain
    true_block = test_df.iloc[test_idx:test_idx + max_horizon].copy()
    current_train = pd.concat([current_train, true_block], ignore_index=True)
    test_idx += max_horizon

# ensure exact test size
forecasts = forecasts[:test_size]

print(forecasts)