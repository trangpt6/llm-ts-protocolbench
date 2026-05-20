import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values
train_size = 1044
train = target[:train_size]
test = target[train_size:]
forecasts = []
history = list(train)
for i in range(len(test)):
    model = SARIMAX(history, order=(1,1,1), seasonal_order=(1,1,1,52), trend='c')
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=4)
    forecasts.append(pred[0])
    history.append(test[i])
print(forecasts)