import pandas as pd
import numpy as np
import lightgbm as lgb
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

# Preprocessing: Insert missing timestamp 2002-01-06
full_range = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='W-SUN')
df = df.set_index('DATE').reindex(full_range).reset_index().rename(columns={'index': 'DATE'})

# Preprocessing: Interpolate target and fill exogenous missing values
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')
df['AGE 25-49'] = df['AGE 25-49'].fillna(0)
df['AGE 50-64'] = df['AGE 50-64'].fillna(0)
df = df.fillna(0)

# Define split
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

target_col = '% WEIGHTED ILI'
hyperparams = {'n_estimators': 50, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42}
lags = 13
horizon = 4

def create_features(data, target_name, num_lags):
    df_feat = data.copy()
    for i in range(1, num_lags + 1):
        df_feat[f'lag_{i}'] = df_feat[target_name].shift(i)
    return df_feat.dropna()

# Rolling forecast with retraining
all_forecasts = []
# We need to generate forecasts for the entire test set.
# Since we move 1 step at a time and predict 4 steps ahead, 
# we collect the first prediction of each 4-step forecast to cover the test set.
for i in range(test_size):
    current_train_end = train_size + i
    train_data = df.iloc[:current_train_end]
    
    # Feature engineering for training
    train_feat = create_features(train_data[[target_col]], target_col, lags)
    X_train = train_feat.drop(columns=[target_col])
    y_train = train_feat[target_col]
    
    # Retrain model
    model = lgb.LGBMRegressor(**hyperparams)
    model.fit(X_train, y_train)
    
    # Recursive multi-step forecast
    current_window = list(df.iloc[current_train_end - lags:current_train_end][target_col].values)
    
    # Predict 4 steps ahead
    preds = []
    temp_window = current_window.copy()
    for h in range(horizon):
        X_test = np.array(temp_window[-lags:]).reshape(1, -1)
        y_pred = model.predict(X_test)[0]
        preds.append(y_pred)
        temp_window.append(y_pred)
    
    # In a rolling scenario with horizon 4 and advancement 1, 
    # to cover the test set we take the prediction for the immediate next step.
    all_forecasts.append(preds[0])

print(all_forecasts)