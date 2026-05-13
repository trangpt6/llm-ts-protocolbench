import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import random

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
df = df.asfreq('MS')

# no preprocessing needed

# chronological split: 80% train, 20% test
n = len(df)
train_size = int(0.8 * n)
train = df.iloc[:train_size]['Passengers']
test = df.iloc[train_size:]['Passengers']

# fixed model and hyperparameters
model = ExponentialSmoothing(
    endog=train,
    trend='add',
    seasonal='mul',
    seasonal_periods=12,
    damped_trend=False
)
fit = model.fit()

# one-step-ahead recursive forecasting without ground truth updates means
# we can directly produce h-step ahead forecasts from the fitted model
h = len(test)
forecasts = fit.forecast(h).tolist()

print(forecasts)