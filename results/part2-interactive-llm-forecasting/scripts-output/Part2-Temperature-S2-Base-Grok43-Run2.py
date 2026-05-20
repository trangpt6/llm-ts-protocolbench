import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)
target = df['Daily minimum temperatures'].values.astype(float)
for i in range(1, len(target)-1):
    if target[i] == 0:
        target[i] = (target[i-1] + target[i+1]) / 2
train_size = 2920
train = target[:train_size].tolist()
test = target[train_size:]
forecasts = []
current_train = train[:]
for i in range(len(test)):
    model = SARIMAX(current_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 365), trend='n')
    fitted_model = model.fit(disp=False)
    pred = fitted_model.forecast(steps=1)[0]
    forecasts.append(pred)
    current_train.append(test[i])
print(forecasts)