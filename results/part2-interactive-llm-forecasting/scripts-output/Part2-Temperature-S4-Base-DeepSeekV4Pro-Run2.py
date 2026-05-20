import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date')
series = df.set_index('Date')['Daily minimum temperatures']

# Split according to Turn 0
train_size = int(0.8 * len(series))
train = series.iloc[:train_size]
test = series.iloc[train_size:]

# Fixed hyperparameters from Turn 2
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 365)
trend = 'n'   # no intercept

# Block-wise rolling forecast
block_size = 30
h = block_size
forecasts = []

current_train = train.copy()
for i in range(0, len(test), block_size):
    end_idx = min(i + block_size, len(test))
    steps = end_idx - i
    
    # Fit SARIMA on current training data
    model = SARIMAX(current_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend=trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fit_res = model.fit(disp=False)
    
    # Forecast the required horizon
    pred = fit_res.forecast(steps=steps)
    forecasts.extend(pred.tolist())
    
    # Update training data with actual values of this block
    actual_block = test.iloc[i:end_idx]
    current_train = pd.concat([current_train, actual_block])

# Print final forecast list
print(forecasts)