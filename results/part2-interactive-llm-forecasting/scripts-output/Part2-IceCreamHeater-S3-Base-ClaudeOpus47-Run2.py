import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
y = df['Ice cream'].astype(float).values

n = len(y)
train_size = int(0.8 * n)
train = y[:train_size]
test = y[train_size:]

H = 12
forecasts = []
history = list(train)

for i in range(len(test)):
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.forecast(steps=H)
    forecasts.append(float(np.asarray(pred)[0]))
    history.append(float(test[i]))

print(forecasts)