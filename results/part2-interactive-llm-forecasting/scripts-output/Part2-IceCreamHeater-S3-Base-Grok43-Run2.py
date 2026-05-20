import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = df['Ice cream'].values
train_size = 158
train = target[:train_size]
test = target[train_size:]
forecasts = []
current_train = list(train)
for i in range(len(test)):
    model = SARIMAX(current_train, order=(1,1,1), seasonal_order=(1,1,1,12), trend='c')
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=12)
    forecasts.append(float(pred[0]))
    current_train.append(test[i])
print(forecasts)