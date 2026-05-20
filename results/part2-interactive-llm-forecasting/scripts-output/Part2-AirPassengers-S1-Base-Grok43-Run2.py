import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
train = df.iloc[:115]['Passengers']
model = ExponentialSmoothing(train, damped_trend=False, seasonal='mul', seasonal_periods=12, trend='add')
fitted = model.fit()
forecasts = fitted.forecast(steps=29).tolist()
print(forecasts)