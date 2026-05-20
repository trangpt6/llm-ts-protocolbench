import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# Read CSV
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing: drop rows with TOTAL PATIENTS == 0 (summer placeholder)
df = df[df['TOTAL PATIENTS'] != 0].copy()

# Drop completely empty AGE 25-49 column if present
if 'AGE 25-49' in df.columns:
    df.drop(columns=['AGE 25-49'], inplace=True)

# Ensure DATE column is datetime and sorted
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract target series
target_col = '% WEIGHTED ILI'
series = df[target_col].astype(float)

# Chronological split: 80% train, 20% test
total_timesteps = len(series)
train_size = int(0.8 * total_timesteps)
train = series.iloc[:train_size]
test = series.iloc[train_size:]

# Forecasting setup: block-wise rolling, block size = 52, horizon = 52
block_size = 52
forecast_horizon = 52
test_len = len(test)

forecasts = []

# Initialize training data as the original training portion
current_train = train.copy()

# How many full blocks of 52 and remainder
num_full_blocks = test_len // block_size
remainder = test_len % block_size

# Iteratively forecast blocks
for i in range(num_full_blocks):
    # Fit SARIMA on current training data
    model = SARIMAX(current_train,
                    order=(2, 0, 2),
                    seasonal_order=(1, 1, 1, 52),
                    trend='n',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    
    # Forecast next 52 steps
    pred = fitted.forecast(steps=block_size)
    forecasts.extend(pred.tolist())
    
    # Update training data with true values from this block
    block_start_idx = i * block_size
    block_end_idx = block_start_idx + block_size
    true_block = test.iloc[block_start_idx:block_end_idx]
    current_train = pd.concat([current_train, true_block])

# If remainder exists, forecast remaining steps
if remainder > 0:
    model = SARIMAX(current_train,
                    order=(2, 0, 2),
                    seasonal_order=(1, 1, 1, 52),
                    trend='n',
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    pred = fitted.forecast(steps=remainder)
    forecasts.extend(pred.tolist())

print(forecasts)