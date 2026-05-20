import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df['Ice cream'].tolist()
train_size = 158
history = data[:train_size]
test = data[train_size:]
forecasts = []
for i in range(len(test)):
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    results = model.fit(disp=False)
    pred = results.forecast(steps=1)[0]
    forecasts.append(pred)
    history.append(test[i])
print(forecasts)