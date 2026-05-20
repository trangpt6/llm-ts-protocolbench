import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
data = pd.read_csv(r'../../../data/AirPassengers.csv')
data['Month'] = pd.to_datetime(data['Month'])
data.set_index('Month', inplace=True)
target = data['Passengers']
train_size = 115
train = target.iloc[:train_size]
test = target.iloc[train_size:]
forecasts = []
current_train = train.copy()
for i in range(len(test)):
    model = SARIMAX(current_train, order=(1,1,1), seasonal_order=(1,1,1,12), trend='n')
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=12)
    forecasts.append(float(pred.iloc[0]))
    next_val = test.iloc[i]
    new_index = test.index[i]
    current_train = pd.concat([current_train, pd.Series([next_val], index=[new_index])])
print(forecasts)