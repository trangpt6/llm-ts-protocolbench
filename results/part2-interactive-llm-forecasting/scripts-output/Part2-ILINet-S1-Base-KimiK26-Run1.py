import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset from local file
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

# minimal preprocessing: forward fill missing values in non-target columns only
target_col = '% WEIGHTED ILI'
non_target_cols = [c for c in df.columns if c not in ['DATE', target_col]]
df[non_target_cols] = df[non_target_cols].ffill()

# strict chronological 80/20 split defined in turn 0
train_size = 1044
train = df.iloc[:train_size]
test = df.iloc[train_size:]
train_endog = train[target_col]

# fit SARIMA with fixed hyperparameters on training data once
model = SARIMAX(train_endog, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c')
results = model.fit(disp=False)

# recursive one-step-ahead forecasting over full test set without retraining or true values
forecasts = []
current_results = results
for _ in range(len(test)):
    pred = current_results.forecast(steps=1).iloc[0]
    forecasts.append(float(pred))
    current_results = current_results.append([pred], refit=False)

print(forecasts)