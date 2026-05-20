import random
import numpy as np
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
target = df['Ice cream']
train = target.iloc[:158]
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
fitted_model = model.fit(disp=False)
forecasts = fitted_model.forecast(steps=40).tolist()
print(forecasts)