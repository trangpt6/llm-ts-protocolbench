import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
y = df['Passengers']
train = y.iloc[:115]
model = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=12, damped_trend=False)
result = model.fit()
forecasts = result.forecast(steps=29).tolist()
print(forecasts)