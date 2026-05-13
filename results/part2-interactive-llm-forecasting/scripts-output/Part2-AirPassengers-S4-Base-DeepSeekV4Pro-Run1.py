import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Reproducibility (not strictly needed but safe)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
series = df['Passengers'].astype(float)

# Chronological split
train_size = int(0.8 * len(series))
train = series.iloc[:train_size]
test = series.iloc[train_size:]

forecasts = []
current_train = train.copy()
pos = 0
block_size = 12
horizon = 12

while len(forecasts) < len(test):
    # Fit model on current training data
    model = ExponentialSmoothing(
        current_train,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    )
    fitted = model.fit()

    # Forecast next horizon steps
    pred = fitted.forecast(horizon)

    # Number of predictions still needed to cover test set
    needed = len(test) - len(forecasts)
    take = min(horizon, needed)
    forecasts.extend(pred[:take].tolist())

    # Obtain ground truth for this block
    actual_block = test.iloc[pos:pos + take]
    # Update training data
    current_train = pd.concat([current_train, actual_block])
    # Move pointer
    pos += take

print(forecasts)