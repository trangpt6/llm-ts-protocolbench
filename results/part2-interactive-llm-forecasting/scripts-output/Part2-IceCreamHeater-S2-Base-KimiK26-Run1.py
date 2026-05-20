import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seed for reproducibility
np.random.seed(42)

# read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# preprocessing from Turn 1: no preprocessing needed, use raw data as-is
# confirm target column and time index
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
df = df.sort_index()

# extract target series
y = df['Ice cream']

# define train/test split: first 157 observations for training
train_size = 157
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# fixed hyperparameters from Turn 2
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

# initialize forecasts list
forecasts = []

# rolling one-step ahead forecasting with retraining at each step
# start with initial training data
current_train = y_train.copy()

for i in range(len(y_test)):
    # fit SARIMA model on current training data
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)
    
    # one-step ahead forecast
    forecast = fitted_model.forecast(steps=1)
    pred = forecast.iloc[0]
    forecasts.append(pred)
    
    # update training data with actual observed value (ground truth enabled)
    actual_value = y_test.iloc[i]
    current_train = pd.concat([current_train, pd.Series([actual_value], index=[y_test.index[i]])])

# print final forecast list
print(forecasts)