import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import random

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month')
df.set_index('Month', inplace=True)

# target
target = 'Ice cream'
y = df[target].astype(float)

# train/test split (chronological, first 158 obs train)
train_size = 158
y_train = y.iloc[:train_size].copy()
y_test = y.iloc[train_size:].copy()

# fixed model parameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

# rolling one-step-ahead forecasting with retraining
forecasts = []
current_train = y_train.copy()

for i in range(len(y_test)):
    # fit model on current training data
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend)
    fit = model.fit(disp=False)
    # forecast next step
    pred = fit.get_forecast(steps=1).predicted_mean.iloc[0]
    forecasts.append(round(pred, 2))  # store rounded forecast
    # update training data with true observation
    current_train = pd.concat([current_train, pd.Series([y_test.iloc[i]], index=[y_test.index[i]])])

# output the final forecast list
print(forecasts)