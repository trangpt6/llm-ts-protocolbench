import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# random seed set for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values
train_size = 1044
train = list(target[:train_size])
test = target[train_size:]
forecasts = []
for i in range(len(test)):
    model = SARIMAX(train, order=(2, 0, 2), seasonal_order=(1, 1, 1, 52), trend='n', enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(pred)
    train.append(test[i])
print(forecasts)