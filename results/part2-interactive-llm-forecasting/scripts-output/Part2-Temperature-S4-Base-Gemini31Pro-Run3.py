import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Ignore warnings to keep output clean
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing steps
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index.freq = 'D'

# Target column
target_col = 'Daily minimum temperatures'
y = df[target_col]

# Train/test split
train_size = int(0.8 * len(y))
train = y.iloc[:train_size]
test = y.iloc[train_size:]

# Setup forecasting parameters
block_size = 30
forecasts = []
history = train.copy()

# Block-wise rolling update with retraining
for i in range(0, len(test), block_size):
# Determine the end of the current block
    end_idx = min(i + block_size, len(test))
    current_block_size = end_idx - i
    
# Define and train the SARIMA model
    model = SARIMAX(
        history,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 365),
        trend='n'
    )
    fitted_model = model.fit(disp=False)
    
# Forecast the current block
    block_forecast = fitted_model.forecast(steps=current_block_size)
    forecasts.extend(block_forecast.tolist())
    
# Update history with the true values of the current block for the next retraining
    history = pd.concat([history, test.iloc[i:end_idx]])

# Print final forecasts
print(forecasts)