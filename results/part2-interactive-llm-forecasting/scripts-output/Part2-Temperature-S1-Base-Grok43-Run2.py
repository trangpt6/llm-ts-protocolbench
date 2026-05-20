import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = 'Daily minimum temperatures'
train_size = 2920
train_data = df[target].iloc[:train_size]
non_zero_median = train_data[train_data != 0].median()
df[target] = df[target].replace(0, non_zero_median)
train = df[target].iloc[:train_size].values
model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 365), trend='n')
fitted_model = model.fit(disp=False)
forecasts = fitted_model.forecast(steps=730).tolist()
print(forecasts)