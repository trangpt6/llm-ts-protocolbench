import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set random seed for reproducibility
np.random.seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# Preprocessing: parse Month as datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').asfreq('MS')

# Primary target
target = df['Passengers']

# Chronological train/test split
train_size = int(0.8 * len(target))  # 115
train = target.iloc[:train_size]
test = target.iloc[train_size:]

# Forecasting setup
block_size = 12
horizon = 12

# Block-wise rolling update with retraining
forecasts = []
train_data = train.copy()

i = 0
while i < len(test):
    # Fit the model on current training data
    model = ExponentialSmoothing(
        train_data,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    )
    fitted = model.fit()
    
    # Forecast horizon steps
    preds = fitted.forecast(horizon)
    
    # Determine how many predictions we actually need for this block
    steps_needed = min(block_size, len(test) - i)
    block_forecasts = preds[:steps_needed]
    forecasts.extend(block_forecasts.tolist())
    
    # Update training data with ground truth from this block
    actuals = test.iloc[i:i+steps_needed]
    train_data = pd.concat([train_data, actuals])
    
    i += steps_needed

# Output final forecasts as a flat Python list
print(forecasts)