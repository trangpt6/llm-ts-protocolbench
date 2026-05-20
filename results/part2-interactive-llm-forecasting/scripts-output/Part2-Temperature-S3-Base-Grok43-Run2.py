import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seed for reproducibility
np.random.seed(42)
data = pd.read_csv(r'../../../data/Temperature.csv')
target = data['Daily minimum temperatures'].values.astype(float)
train_size = 2920
train = target[:train_size]
test = target[train_size:]
forecasts = []
current_train = list(train)
for i in range(len(test)):
    model = SARIMAX(current_train, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n', enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=7)
    forecasts.append(float(pred[0]))
    current_train.append(test[i])
print(forecasts)