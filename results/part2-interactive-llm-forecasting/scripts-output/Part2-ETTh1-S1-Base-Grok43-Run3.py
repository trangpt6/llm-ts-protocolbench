import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
np.random.seed(42)
df = pd.read_csv(r'../../../data/ETTh1.csv')
train_size = 13936
train = df['OT'].iloc[:train_size]
model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24), trend='c')
fitted = model.fit(disp=False)
forecasts = fitted.forecast(steps=len(df) - train_size).tolist()
print(forecasts)