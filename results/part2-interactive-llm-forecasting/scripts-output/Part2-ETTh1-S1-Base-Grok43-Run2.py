import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
np.random.seed(42)
df = pd.read_csv(r'../../../data/ETTh1.csv')
train_size = int(0.8 * len(df))
train = df.iloc[:train_size]['OT']
test_len = len(df) - train_size
model = SARIMAX(train, order=(2,1,2), seasonal_order=(1,1,1,24), trend='c')
fitted = model.fit(disp=False)
forecasts = fitted.forecast(steps=test_len).tolist()
print(forecasts)