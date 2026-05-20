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

# Read data and apply preprocessing
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Select primary target
target = df['Ice cream']

# Train/test split
train_size = 158
train_data = target.iloc[:train_size]
test_data = target.iloc[train_size:]

# Initialize history and forecasts list
history = train_data.copy()
forecasts = []

# Block-wise rolling update setup
test_len = len(test_data)
block_size = 12

# Forecasting loop
for i in range(0, test_len, block_size):
    current_block_size = min(block_size, test_len - i)
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    fit_model = model.fit(disp=False)
    pred = fit_model.forecast(steps=current_block_size)
    forecasts.extend(pred.tolist())
    history = pd.concat([history, test_data.iloc[i:i+current_block_size]])

# Print final forecasts
print(forecasts)