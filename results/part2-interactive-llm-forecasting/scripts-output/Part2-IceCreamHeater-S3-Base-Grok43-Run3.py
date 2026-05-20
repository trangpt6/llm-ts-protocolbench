import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
target = df['Ice cream']
train_size = 158
train = target.iloc[:train_size]
test = target.iloc[train_size:]
forecasts = []
current_train = train.tolist()
for i in range(len(test)):
    model = SARIMAX(current_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    fitted_model = model.fit(disp=False)
    pred = fitted_model.forecast(steps=12)
    forecasts.append(pred[0])
    current_train.append(test.iloc[i])
print(forecasts)