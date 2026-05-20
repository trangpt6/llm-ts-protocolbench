import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seed for reproducibility
np.random.seed(42)

# read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# preprocessing: parse Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# ensure chronological ordering
df = df.sort_values('Date').reset_index(drop=True)

# set Date as index
df.set_index('Date', inplace=True)

# extract target series
y = df['Daily minimum temperatures']

# apply fixed train/test split
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

# generate one-step ahead forecasts recursively over test set
# no retraining, no use of ground truth after each prediction
forecasts = []
history = y_train.copy()

for i in range(len(y_test)):
    # forecast one step ahead
    forecast = fitted_model.forecast(steps=1)
    pred = forecast.iloc[0]
    forecasts.append(float(pred))
    
    # update history with predicted value (not true value)
    history = pd.concat([history, pd.Series([pred], index=[y_test.index[i]])])
    
    # re-extend the model with new observation for next forecast
    fitted_model = fitted_model.extend([pred])

print(forecasts)