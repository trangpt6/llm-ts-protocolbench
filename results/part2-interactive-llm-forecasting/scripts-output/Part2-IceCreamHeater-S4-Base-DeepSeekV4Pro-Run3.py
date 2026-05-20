import pandas as pd
import numpy as np
import random
import statsmodels.api as sm

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
series = df['Ice cream'].values

# Chronological split
train_size = 158
train_series = series[:train_size]
test_series = series[train_size:]

# Initialize history with training data
history = list(train_series)
forecasts = []
block_size = 12
horizon = 12

# Block-wise rolling forecast with retraining after each block
idx = 0
while idx < len(test_series):
    current_needed = min(block_size, len(test_series) - idx)
    # Fit SARIMA on current history
    model = sm.tsa.SARIMAX(history, order=(1,1,1), seasonal_order=(1,1,1,12), trend='c')
    res = model.fit(disp=False)
    # Produce horizon-step forecast
    preds = res.forecast(steps=horizon)
    # Keep only the needed predictions for this block
    forecasts.extend(preds[:current_needed])
    # Update history with ground truth of this block
    history.extend(test_series[idx:idx+current_needed])
    idx += current_needed

print(forecasts)