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
series = df['Passengers'].astype(float).values

n = len(series)
train_size = int(0.8 * n)

forecasts = []
for i in range(train_size, n):
    history = series[:i]
    model = ExponentialSmoothing(
        history,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False,
    )
    fit = model.fit(optimized=True)
    yhat = fit.forecast(1)[0]
    forecasts.append(float(yhat))

print(forecasts)