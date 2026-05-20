import random
random.seed(42)
import numpy as np
np.random.seed(42)

import pandas as pd
import lightgbm as lgb

# Read the data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])

# Drop the completely empty column as decided in Turn 1
df = df.drop(columns=['AGE 25-49'])

# Use DATE as index
df = df.set_index('DATE')
df = df.sort_index()

target_col = '% WEIGHTED ILI'
lags = 13

# Split into train/test sizes exactly as in Turn 0
train_size = int(0.8 * len(df))
train_end_idx = train_size - 1  # last training index

# Extract target and exogenous features
target = df[target_col].astype(float)
exog = df.drop(columns=[target_col]).astype(float)

# Convert to lists for efficient appending later
target_list = target.tolist()
exog_list = exog.values.tolist()

# Helper to build a feature vector at time t using available data lists up to t
def build_features(t, available_target, available_exog):
    feats = []
    for lag in range(1, lags + 1):
        feats.append(available_target[t - lag])
    feats.extend(available_exog[t])
    return np.array(feats)

# Prepare initial training data: pairs (features at time i, target[i+1]) for i in [lags-1, train_end_idx-1]
X_train_list = []
y_train_list = []
for i in