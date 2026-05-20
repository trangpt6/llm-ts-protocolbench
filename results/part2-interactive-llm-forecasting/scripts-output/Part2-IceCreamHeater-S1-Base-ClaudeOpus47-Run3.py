import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.set_index('Month').asfreq('MS')

y = df['Ice cream'].astype(float)
n = len(y)
train_size = int(0.8 * n)
train = y.iloc[:train_size]
test = y.iloc[train_size:]

model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
fit = model.fit(disp=False)

forecasts = fit.forecast(steps=len(test)).tolist()

print(forecasts)