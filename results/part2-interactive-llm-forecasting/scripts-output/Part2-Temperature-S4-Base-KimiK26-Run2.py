import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

y = df['Daily minimum temperatures'].values
n = len(y)
train_size = int(0.8 * n)
train_data = y[:train_size]
test_data = y[train_size:]

forecasts = []
block_size = 30

for start in range(0, len(test_data), block_size):
    end = min(start + block_size, len(test_data))
    steps = end - start
    model = SARIMAX(train_data, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n')
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=steps)
    forecasts.extend(pred.tolist())
    train_data = np.concatenate([train_data, test_data[start:end]])

print(forecasts)