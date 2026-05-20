import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].astype(float).values
for i in range(len(target)):
    if target[i] == 0:
        non_zero = []
        j = i - 1
        while len(non_zero) < 7 and j >= 0:
            if target[j] != 0:
                non_zero.append(target[j])
            j -= 1
        if len(non_zero) > 0:
            target[i] = np.median(non_zero)
train_size = 2920
train = target[:train_size]
test = target[train_size:]
forecasts = []
current_train = list(train)
for i in range(len(test)):
    model = SARIMAX(current_train, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n')
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=7)
    forecasts.append(float(pred[0]))
    current_train.append(test[i])
print(forecasts)