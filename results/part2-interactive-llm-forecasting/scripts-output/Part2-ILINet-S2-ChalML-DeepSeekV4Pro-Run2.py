import random
import numpy as np
import pandas as pd
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

# Load and preprocess
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'], index_col='DATE')[['% WEIGHTED ILI']]
df = df.sort_index()

# Reindex to regular weekly frequency (Sunday)
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_idx)

# Linear interpolation for missing target values (only the two known gaps)
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

series = df['% WEIGHTED ILI']

# Train/test split
train_size = 1050
train_series = series.iloc[:train_size]
test_series = series.iloc[train_size:]

# Rolling one-step ahead forecasting with retraining at each step
known = train_series.tolist()
lags = 13
forecasts = []

for i in range(len(test_series)):
    t = len(known)  # current size of known series, index of next prediction
    
    # Build training data from all known observations (with lags)
    X_train = []
    y_train = []
    for j in range(lags, t):
        X_train.append(known[j-lags:j])
        y_train.append(known[j])
    
    # Train LightGBM model
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    if X_train:
        model.fit(X_train, y_train)
        pred = model.predict([known[t-lags:t]])[0]
    else:
        pred = known[-1]  # fallback, should not occur as t >= lags after first training point
    
    forecasts.append(pred)
    # Append true test value (ground truth) to known for next step
    known.append(test_series.iloc[i])

print(forecasts)