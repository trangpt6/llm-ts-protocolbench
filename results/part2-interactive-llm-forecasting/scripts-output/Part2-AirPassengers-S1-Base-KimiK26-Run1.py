import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# set random seed for reproducibility
np.random.seed(42)

# read dataset from local csv
df = pd.read_csv(r'../../../data/AirPassengers.csv')

# parse time index and set monthly frequency
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.index.freq = 'MS'

# extract primary target series
y = df['Passengers']

# chronological split: first 115 timesteps for training
n_train = 115
train = y.iloc[:n_train]

# fit ExponentialSmoothing with fixed structural hyperparameters
model = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='mul',
    seasonal_periods=12,
    damped_trend=False
)
fitted_model = model.fit()

# recursively forecast one-step ahead over the entire test set without retraining
forecasts = fitted_model.forecast(steps=len(y) - n_train).tolist()

print(forecasts)