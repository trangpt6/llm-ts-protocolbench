import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
# no preprocessing needed per Turn 1
series = df['Ice cream'].values.astype(float)

# define train/test split: train first 158, test remaining 40
train_size = 158
train = series[:train_size]
test = series[train_size:]

# fixed hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# rolling one-step ahead forecasting with retraining after each prediction
history = list(train)
forecasts = []
for t in range(len(test)):
    # fit SARIMA model on current history
    model = ARIMA(history, order=order, seasonal_order=seasonal_order, trend='c')
    fitted = model.fit()
    # forecast next step
    yhat = fitted.forecast(steps=1)[0]
    forecasts.append(yhat)
    # update history with true observed value for next retraining
    history.append(test[t])

# print the forecasts as a flat list
print(forecasts)