import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
np.random.seed(42)
data = pd.read_csv(r'../../../data/ILINet.csv')
target = data['% WEIGHTED ILI'].values
train_size = 1044
train = target[:train_size].tolist()
test = target[train_size:]
forecasts = []
current_train = train.copy()
for i in range(len(test)):
    model = SARIMAX(current_train, order=(1,1,1), seasonal_order=(1,1,1,52), trend='c', enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=4)
    forecasts.append(float(pred[0]))
    current_train.append(test[i])
print(forecasts)