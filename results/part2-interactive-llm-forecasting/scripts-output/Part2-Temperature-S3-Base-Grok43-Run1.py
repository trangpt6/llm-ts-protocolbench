import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values
train_size = 2920
train = target[:train_size]
test = target[train_size:]
forecasts = []
current_train = list(train)
for i in range(len(test)):
    model = SARIMAX(current_train, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n')
    fitted_model = model.fit(disp=False)
    pred = fitted_model.forecast(steps=7)
    forecasts.append(float(pred[0]))
    current_train.append(test[i])
print(forecasts)