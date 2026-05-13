import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
ts = df['Passengers']

# Split: train (first 115 observations), test (remaining 29)
train_end_idx = 115
test_indices = range(train_end_idx, len(ts))

forecasts = []
for i, test_idx in enumerate(test_indices):
    # Use all data up to (but not including) the test point
    history = ts[:test_idx]
    model = ExponentialSmoothing(
        history,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    ).fit()
    fore = model.forecast(1)
    forecasts.append(fore.iloc[0])

print(forecasts)