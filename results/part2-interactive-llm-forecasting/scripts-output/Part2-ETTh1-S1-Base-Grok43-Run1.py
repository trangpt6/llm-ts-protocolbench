import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
np.random.seed(42)
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
train_size = int(0.8 * len(df))
train = df['OT'][:train_size]
test = df['OT'][train_size:]
model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24), trend='c')
fitted = model.fit(disp=False)
forecasts = []
history = list(train.values)
for i in range(len(test)):
    pred = fitted.forecast(steps=1)
    forecasts.append(pred[0])
    history.append(pred[0])
print(forecasts)