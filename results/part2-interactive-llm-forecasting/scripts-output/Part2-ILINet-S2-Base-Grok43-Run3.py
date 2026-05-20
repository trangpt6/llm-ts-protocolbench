import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE').asfreq('W-SUN', method='ffill')
target = df['% WEIGHTED ILI']
train_size = 1043
train = target.iloc[:train_size]
test = target.iloc[train_size:]
forecasts = []
history = list(train.values)
for i in range(len(test)):
    model = SARIMAX(history, order=(2, 0, 2), seasonal_order=(1, 1, 1, 52), trend='n', enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(float(pred))
    history.append(test.iloc[i])
print(forecasts)