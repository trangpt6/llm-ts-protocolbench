import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
y = df['Daily minimum temperatures'].values.astype(float)

train_size = int(0.8 * len(y))
train = y[:train_size]
test = y[train_size:]

model = SARIMAX(train, order=(1,0,1), seasonal_order=(1,1,1,365), trend='n')
fit = model.fit(disp=False)

forecast_result = fit.get_forecast(steps=len(test))
forecasts = forecast_result.predicted_mean.tolist()

print(forecasts)