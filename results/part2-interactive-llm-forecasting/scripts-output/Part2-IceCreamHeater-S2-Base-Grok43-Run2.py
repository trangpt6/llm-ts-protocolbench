import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = df['Ice cream'].values
train_size = 158
train = target[:train_size]
test = target[train_size:]
forecasts = []
history = list(train)
for i in range(len(test)):
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    fitted_model = model.fit(disp=False)
    pred = fitted_model.forecast(steps=1)[0]
    forecasts.append(pred)
    history.append(test[i])
print(forecasts)