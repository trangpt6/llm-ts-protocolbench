import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
series = df['Passengers'].astype(float).tolist()
train_size = 115
current_train = series[:train_size]
test = series[train_size:]
forecasts = []
for i in range(len(test)):
    model = ExponentialSmoothing(current_train, damped_trend=False, seasonal='mul', seasonal_periods=12, trend='add')
    fit = model.fit()
    pred = fit.forecast(1)[0]
    forecasts.append(pred)
    current_train.append(test[i])
print(forecasts)