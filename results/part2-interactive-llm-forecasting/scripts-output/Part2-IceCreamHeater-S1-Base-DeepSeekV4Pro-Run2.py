import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set seed for reproducibility
np.random.seed(42)

# read raw data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# preprocessing: parse Month as datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').asfreq('MS')
target = df['Ice cream'].astype(float)

# chronological split: first 158 train, last 40 test
train = target.iloc[:158]

# fixed model: SARIMA with given hyperparameters
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False
)

# fit model
fit_result = model.fit(disp=False)

# one-step ahead recursive forecast over entire test horizon (40 steps)
forecast_result = fit_result.get_forecast(steps=40)
forecasts = forecast_result.predicted_mean.tolist()

# produce final plain list
print(forecasts)