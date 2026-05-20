import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].tolist()
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]
forecasts = []
current_train = train_data[:]
for i in range(len(test_data)):
    model = ExponentialSmoothing(current_train, trend='add', seasonal='mul', seasonal_periods=12, damped_trend=False)
    fitted_model = model.fit()
    pred = fitted_model.forecast(steps=1)[0]
    forecasts.append(pred)
    current_train.append(test_data[i])
print(forecasts)