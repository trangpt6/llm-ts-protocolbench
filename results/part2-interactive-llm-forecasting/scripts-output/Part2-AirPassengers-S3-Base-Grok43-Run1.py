import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
data = df['Passengers']
train_size = 115
train = data.iloc[:train_size]
test = data.iloc[train_size:]
history = train.copy()
forecasts = []
for i in range(len(test)):
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='n')
    results = model.fit(disp=False)
    pred = results.forecast(steps=12)
    forecasts.append(pred.iloc[0])
    history = pd.concat([history, test.iloc[i:i+1]])
print(forecasts)