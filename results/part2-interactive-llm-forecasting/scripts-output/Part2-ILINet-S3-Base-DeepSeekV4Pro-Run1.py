import pandas as pd
import numpy as np
from pmdarima.arima import ARIMA
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True, errors='ignore')

y = df['% WEIGHTED ILI'].values.astype(float)
n = len(y)
train_size = int(0.8 * n)

forecasts = []
for t in range(n - train_size):
    current_train = y[:train_size + t]
    model = ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c')
    model.fit(current_train, disp=0)
    fc = model.predict(n_periods=4)
    forecasts.append(fc[0])

print(forecasts)