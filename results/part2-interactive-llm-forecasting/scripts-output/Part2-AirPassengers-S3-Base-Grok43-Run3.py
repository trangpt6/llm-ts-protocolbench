import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
data = df['Passengers'].values
train_size = 115
train_data = data[:train_size].tolist()
test_data = data[train_size:].tolist()
forecasts = []
current_data = train_data.copy()
for i in range(len(test_data)):
    model = SARIMAX(current_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='n')
    results = model.fit(disp=False)
    pred = results.forecast(steps=12)
    forecasts.append(float(pred[0]))
    current_data.append(test_data[i])
print(forecasts)