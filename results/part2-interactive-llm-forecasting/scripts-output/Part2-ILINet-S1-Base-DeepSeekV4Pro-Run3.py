import pandas as pd
import numpy as np
from pmdarima import ARIMA

np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df = df.drop(columns=['AGE 25-49', 'AGE 25-64'])
target = df['% WEIGHTED ILI'].astype(float)

train_ts = target.iloc[:1050]

model = ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c', suppress_warnings=True)
model.fit(train_ts)

forecasts = model.predict(n_periods=263)
print(forecasts.tolist())