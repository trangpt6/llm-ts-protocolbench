import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset from local file
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# parse time index
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
df = df.sort_index()

# extract target series
y = df['Passengers']

# chronological split: first 115 observations as training set
train_size = 115
train = y.iloc[:train_size]

# fit ExponentialSmoothing with fixed structural hyperparameters
model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='mul',
    seasonal_periods=12,
    damped_trend=False
)
fitted_model = model.fit()

# generate recursive one-step ahead forecasts over the test set without retraining or ground truth
# forecast the full test length recursively using the fitted model states
forecasts = fitted_model.forecast(steps=len(y) - train_size).tolist()

# ensure plain Python float types
forecasts = [float(x) for x in forecasts]

print(forecasts)