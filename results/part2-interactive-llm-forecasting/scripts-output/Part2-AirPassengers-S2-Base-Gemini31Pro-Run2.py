import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

train_size = 115
train = list(y[:train_size])
test = list(y[train_size:])

forecasts = []
history = train.copy()

for i in range(len(test)):
    model = ExponentialSmoothing(
        history,
        trend='add',
        damped_trend=False,
        seasonal='mul',
        seasonal_periods=12
    )
    fit_model = model.fit()
    pred = fit_model.forecast(1)[0]
    forecasts.append(pred)
    history.append(test[i])

print(forecasts)