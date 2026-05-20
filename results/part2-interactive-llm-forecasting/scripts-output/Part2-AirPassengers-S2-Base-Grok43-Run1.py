import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df = df.asfreq('MS')
train_size = 115
history = df.iloc[:train_size].copy()
test = df.iloc[train_size:]
forecasts = []
for i in range(len(test)):
    model = ExponentialSmoothing(history['Passengers'], damped_trend=False, seasonal='mul', seasonal_periods=12, trend='add')
    fit = model.fit()
    pred = fit.forecast(1)
    forecasts.append(float(pred.iloc[0]))
    next_date = test.index[i]
    next_value = test.loc[next_date, 'Passengers']
    new_row = pd.DataFrame({'Passengers': [next_value]}, index=[next_date])
    history = pd.concat([history, new_row])
print(forecasts)