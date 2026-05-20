import pandas as pd
import numpy as np
import random
import statsmodels.api as sm

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], dayfirst=False)
df = df.set_index('Date').sort_index()
target_col = 'Daily minimum temperatures'

# total row count
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

# split chronologically
train = df.iloc[:train_size][target_col]
test = df.iloc[train_size:][target_col]

# fit SARIMA model
model = sm.tsa.SARIMAX(train,
                       order=(1, 0, 1),
                       seasonal_order=(1, 1, 1, 365),
                       trend='n',
                       enforce_stationarity=False,
                       enforce_invertibility=False)
fit = model.fit(disp=False)

# recursive one-step ahead forecasting
forecasts = []
current_model = fit
for i in range(len(test)):
    pred = current_model.forecast(steps=1)[0]
    forecasts.append(pred)
    current_model = current_model.append([pred], refit=False)

print(forecasts)