import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
target_col = '% WEIGHTED ILI'

# Preprocessing: Handle missing timestamp 2002-01-06
df = df.set_index('DATE').asfreq('W-SUN')
df[target_col] = df[target_col].interpolate(method='linear')

# Preprocessing: Fill missing exogenous columns with 0
exo_cols = ['AGE 25-49', 'AGE 50-64']
for col in exo_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Define split point based on Turn 0
train_size = 1044
test_size = 261
full_data = df[target_col].values
train_data = list(full_data[:train_size])
test_data = full_data[train_size:]

# SARIMA parameters from Turn 2
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

# Forecasting setup: Block-wise rolling update
block_size = 52
forecasts = []
current_train = list(train_data)

# Iterate through test set in blocks
for i in range(0, test_size, block_size):
    # Determine current block horizon
    current_horizon = min(block_size, test_size - i)
    
    # Train model on current training set
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # Forecast multi-step ahead for the block
    block_forecast = model_fit.forecast(steps=current_horizon)
    forecasts.extend(block_forecast.tolist())
    
    # Update training set with ground truth for the next block (retraining rule)
    if i + current_horizon < test_size:
        actual_block = test_data[i:i+current_horizon]
        current_train.extend(actual_block)

# Ensure output is exactly the test size
final_forecasts = forecasts[:test_size]
print(final_forecasts)