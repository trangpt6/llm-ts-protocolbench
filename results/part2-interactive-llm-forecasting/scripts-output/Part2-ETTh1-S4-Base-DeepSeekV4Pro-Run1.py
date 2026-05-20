import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seeds for reproducibility
np.random.seed(42)

# read the dataset
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')

# preprocessing: replace constant-value blocks (all 24 hours identical) with NaN
df_processed = df.copy()
daily_groups = df_processed.groupby(df_processed.index.date)
for date, group in daily_groups:
    if (group['OT'].nunique() == 1) and (len(group) == 24):
        df_processed.loc[group.index, 'OT'] = np.nan

# chronological split
total_timesteps = len(df_processed)
train_size = int(0.8 * total_timesteps)
train = df_processed.iloc[:train_size]
test = df_processed.iloc[train_size:]

# forecast parameters
block_size = 168
horizon = 168

# initial training series
train_series = train['OT'].copy()

# list to store all forecasts
forecasts = []

# block-wise rolling update
remaining_test = len(test)
start = 0
while remaining_test > 0:
    # fit SARIMA on current training data
    model = SARIMAX(train_series, order=(2,1,2), seasonal_order=(1,1,1,24), trend='n',
                    missing='drop', enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    
    # determine how many steps to forecast
    steps = min(block_size, remaining_test)
    
    # forecast
    pred = fit.forecast(steps=steps)
    forecasts.extend(pred.tolist())
    
    # update training data with true test values for this block
    true_values = test['OT'].iloc[start:start+steps]
    train_series = pd.concat([train_series, true_values])
    
    # move window
    start += steps
    remaining_test -= steps

# output final forecast list
print(forecasts)