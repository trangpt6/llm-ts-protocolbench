import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset from local file
df = pd.read_csv(r'../../../data/Temperature.csv')

# parse dates and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# preprocessing: replace clear outlier on 1984-11-26 with average of adjacent days
outlier_date = pd.Timestamp('1984-11-26')
mask = df['Date'] == outlier_date
if mask.any():
    idx = df.index[mask][0]
    prev_val = df.loc[idx - 1, 'Daily minimum temperatures']
    next_val = df.loc[idx + 1, 'Daily minimum