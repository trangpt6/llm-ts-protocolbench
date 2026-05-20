import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set seed for reproducibility
np.random.seed(42)

# read the CSV file
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# no preprocessing (Turn 1)
target = 'Ice cream'
series = df[target].values

# train/test split: first 158 train, next 40 test (Turn 2)
train = series[:158]

# define model with fixed hyperparameters (Turn 2)
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c',
                enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(disp=False)

# generate 40-step dynamic multi-step forecast (no ground truth updates)
forecast = result.get_forecast(steps=40).predicted_mean

# output as flat list
print(forecast.tolist())