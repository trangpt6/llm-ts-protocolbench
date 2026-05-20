import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seed for reproducibility
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# parse dates
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# set index
df.set_index('Date', inplace=True)

# target series
y = df['Daily minimum temperatures']

# chronological 80/20 split
train_size = 2920
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# fixed hyperparameters
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

# fit SARIMA model once on training data
model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order, trend=trend)
fitted_model = model.fit(disp=False)

# recursive one-step ahead forecasting over full test set
# no retraining, no use of ground truth
forecasts = []
last_history = y_train.copy()

for i in range(len(y_test)):
    # forecast one step ahead
    pred = fitted_model.get_forecast(steps=1, exog=None)
    forecast_value = pred.predicted_mean.iloc[0]
    forecasts.append(float(forecast_value))
    
    # update model state with predicted value for next iteration
    # extend the model with the new observation (forecast)
    fitted_model = fitted_model.append([forecast_value], refit=False)

# print final forecast list
print(forecasts)