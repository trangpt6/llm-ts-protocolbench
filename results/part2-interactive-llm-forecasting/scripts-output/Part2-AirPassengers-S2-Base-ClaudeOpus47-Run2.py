import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# reproducibility
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
y = df['Passengers'].astype(float).values

n = len(y)
train_size = int(0.8 * n)

forecasts = []
for i in range(train_size, n):
    history = y[:i]
    model = ExponentialSmoothing(
        history,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False,
        initialization_method='estimated'
    )
    fit = model.fit(optimized=True)
    pred = fit.forecast(1)
    forecasts.append(float(pred[0]))

print(forecasts)