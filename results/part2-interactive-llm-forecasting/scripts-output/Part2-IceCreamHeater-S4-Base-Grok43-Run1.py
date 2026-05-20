import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
data = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = data['Ice cream'].values
train_size = 158
current_train = target[:train_size]
test = target[train_size:]
forecasts = []
i = 0
while i < len(test):
    model = SARIMAX(current_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    fitted_model = model.fit(disp=False)
    h = min(12, len(test) - i)
    pred = fitted_model.forecast(steps=h)
    forecasts.extend(pred.tolist())
    current_train = np.concatenate((current_train, test[i:i+h]))
    i += h
print(forecasts)