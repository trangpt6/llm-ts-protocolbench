import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# set random seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df = df.asfreq('M')
train = df.iloc[:115]['Passengers']
model = ExponentialSmoothing(train, damped_trend=False, seasonal='mul', seasonal_periods=12, trend='add')
fitted_model = model.fit()
forecasts = fitted_model.forecast(steps=29).tolist()
print(forecasts)