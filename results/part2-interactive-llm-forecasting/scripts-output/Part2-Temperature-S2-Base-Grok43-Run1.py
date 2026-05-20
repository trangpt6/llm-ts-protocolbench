import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = 'Daily minimum temperatures'
non_zero = df[target][df[target] != 0]
median_val = non_zero.median()
df[target] = df[target].replace(0, median_val)
train_size = 2920
train = df[target][:train_size].values
test = df[target][train_size:].values
forecasts = []
current_train = list(train)
for i in range(len(test)):
    model = SARIMAX(current_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 365), trend='n')
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(pred)
    current_train.append(test[i])
print(forecasts)