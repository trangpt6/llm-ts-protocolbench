import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
target_col = 'Passengers'
series = df[target_col].astype(float)

# Split into train and test according to Turn 0
n_total = len(series)
train_size = int(0.8 * n_total)  # 115
train = series.iloc[:train_size]
test = series.iloc[train_size:]  # 29 values

# Forecasting setup parameters
block_size = 12
horizon = 12

# Hyperparameters from Turn 2
model_params = {
    'trend': 'add',
    'seasonal': 'mul',
    'seasonal_periods': 12,
    'damped_trend': False
}

# Block-wise rolling update with retraining
forecasts = []
current_train = train.copy()

for start in range(0, len(test), block_size):
    # Fit model on current training data
    model = ExponentialSmoothing(current_train, **model_params)
    fitted = model.fit()
    # Forecast the next horizon steps
    pred = fitted.forecast(horizon)
    # Append the predictions needed for this block
    # Take min(horizon, remaining test observations)
    remaining = len(test) - start
    n_pred = min(horizon, remaining)
    forecasts.extend(pred[:n_pred].tolist())
    # Update training data with true values of this block
    true_block = test.iloc[start: start + block_size]
    current_train = pd.concat([current_train, true_block])

# Print the forecast list as the final output
print(forecasts)