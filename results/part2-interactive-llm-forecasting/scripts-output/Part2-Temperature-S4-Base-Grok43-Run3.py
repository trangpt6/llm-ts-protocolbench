import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# set random seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values
train_size = 2921
train = target[:train_size]
test = target[train_size:]
forecasts = []
current_train = list(train)
block_size = 30
horizon = 30
i = 0
while i < len(test):
    model = SARIMAX(current_train, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n', enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    steps = min(horizon, len(test) - i)
    pred = fitted.forecast(steps=steps)
    forecasts.extend(pred.tolist())
    current_train.extend(test[i:i+steps])
    i += steps
print(forecasts)