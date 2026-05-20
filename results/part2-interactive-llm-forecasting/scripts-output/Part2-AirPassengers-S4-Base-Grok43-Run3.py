import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
data = pd.read_csv(r'../../../data/AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
target = data['Passengers']
train = target.iloc[:115]
test = target.iloc[115:]
forecasts = []
current_train = train.copy()
i = 0
while i < len(test):
    block_size = min(12, len(test) - i)
    model = ExponentialSmoothing(current_train, damped_trend=False, seasonal='mul', seasonal_periods=12, trend='add')
    fitted_model = model.fit()
    pred = fitted_model.forecast(steps=block_size)
    forecasts.extend(pred.tolist())
    current_train = pd.concat([current_train, test.iloc[i:i+block_size]])
    i += block_size
print(forecasts)