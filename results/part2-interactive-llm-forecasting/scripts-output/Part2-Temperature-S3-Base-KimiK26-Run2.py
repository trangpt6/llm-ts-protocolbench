import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seed for reproducibility
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# parse time index and target
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
y = df['Daily minimum temperatures'].values

# chronological split per Turn 0
train_size = 2920
train = y[:train_size]
test = y[train_size:]

forecasts = []
history = train.tolist()

# rolling multi-step forecast: retrain each step, horizon 7, update with true value
for i in range(len(test)):
    model = SARIMAX(history, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n')
    result = model.fit(disp=False)
    f = result.forecast(steps=7)
    forecasts.extend(f.tolist())
    history.append(test[i])

print(forecasts)