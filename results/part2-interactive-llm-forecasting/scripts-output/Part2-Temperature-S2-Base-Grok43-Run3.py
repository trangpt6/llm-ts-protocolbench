import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import random
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)
target = df['Daily minimum temperatures'].copy()
zero_indices = target[target == 0].index.tolist()
for idx in zero_indices:
    prev_values = target.loc[:idx].iloc[-8:-1]
    median_val = prev_values.median()
    target.loc[idx] = median_val
train_size = 2920
train_data = target.iloc[:train_size]
test_data = target.iloc[train_size:]
forecasts = []
current_train = train_data.copy()
for i in range(len(test_data)):
    model = SARIMAX(current_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 365), trend=None)
    fitted_model = model.fit(disp=False)
    pred = fitted_model.forecast(steps=1)
    forecasts.append(float(pred.iloc[0]))
    current_train = pd.concat([current_train, test_data.iloc[i:i+1]])
print(forecasts)