import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import random

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])

# Drop entirely empty column
df = df.drop(columns=['AGE 25-49'])

# Drop rows where all numeric columns are zero (non-reporting weeks)
numeric_cols = df.select_dtypes(include=[np.number]).columns
zero_mask = (df[numeric_cols] == 0).all(axis=1)
df = df[~zero_mask].reset_index(drop=True)

# Chronological split dates
train_end = '2017-11-26'
test_start = '2017-12-03'

train_df = df[df['DATE'] <= train_end].sort_values('DATE')
test_df = df[df['DATE'] >= test_start].sort_values('DATE')

train_target = train_df['% WEIGHTED ILI'].values
test_target = test_df['% WEIGHTED ILI'].values

# Block-wise rolling forecast setup
block_size = 52
forecasts = []
current_train = train_target.copy()
remaining_test = test_target.copy()

# Fit initial model and generate forecast block by block
while len(remaining_test) > 0:
    # Fit SARIMA on current training data
    model = SARIMAX(current_train,
                    order=(2, 0, 2),
                    seasonal_order=(1, 1, 1, 52),
                    trend='n',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    
    # Determine how many steps to forecast in this block
    n_forecast = min(block_size, len(remaining_test))
    
    # Forecast
    pred = fitted.forecast(steps=n_forecast)
    forecasts.extend(pred.tolist())
    
    # Update train with true values for the forecasted block
    current_train = np.concatenate([current_train, remaining_test[:n_forecast]])
    remaining_test = remaining_test[n_forecast:]

# Print final forecast list
print(forecasts)