import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read dataset from local csv
df = pd.read_csv(r'../../../data/AirPassengers.csv', index_col='Month', parse_dates=True)

# define target and split sizes
target = 'Passengers'
train_size = 115

# chronological split
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

forecasts = []

# rolling one-step ahead forecast with retraining at each step
for i in range(len(test)):
    model = ExponentialSmoothing(
        train[target],
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    )
    fitted = model.fit()
    pred = fitted.forecast(1).iloc[0]
    forecasts.append(float(pred))
    train = pd.concat([train, test.iloc[[i]]])

print(forecasts)