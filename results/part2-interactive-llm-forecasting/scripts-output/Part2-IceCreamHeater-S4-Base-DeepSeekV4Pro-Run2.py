import numpy as np
np.random.seed(42)

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read the raw CSV
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# Preprocessing: parse Month as datetime and set index
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Define target
target = 'Ice cream'

# Chronological split: first 158 rows train, rest test
train = df.iloc[:158]
test = df.iloc[158:]

# Initial training series
train_series = train[target].copy()

# Forecasting setup parameters
horizon = 12
test_length = len(test)

# Fixed model hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

forecasts = []
current_train = train_series.copy()
block_start = 0  # index within test

while len(forecasts) < test_length:
    # Fit SARIMA model on current training series
    model = SARIMAX(
        current_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False, maxiter=500, method='bfgs')
    
    # Forecast the next horizon steps
    # If not enough left, we still forecast 12 but will trim later
    fcast = fitted.forecast(steps=horizon)
    
    # Determine how many points to use from this block
    remaining = test_length - len(forecasts)
    n_take = min(horizon, remaining)
    
    # Append the required forecasts
    forecasts.extend(fcast[:n_take].tolist())
    
    # Retrieve true values for the block just forecasted (to update training)
    # Actual true values from test set
    true_block = test[target].iloc[block_start:block_start + n_take]
    
    # Append true values to training series for next retrain
    current_train = pd.concat([current_train, true_block])
    
    # Move block start index
    block_start += n_take

# Output the final forecast list
print(forecasts)