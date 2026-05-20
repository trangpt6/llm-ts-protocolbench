import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# suppress statsmodels warnings to keep output clean
warnings.filterwarnings('ignore')

# load data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
df = df.sort_values('Date')
ts = df['Daily minimum temperatures'].values.astype(float)
n_total = len(ts)

# chronological 80/20 split
n_train = 2921  # int(0.8 * n_total) given 3652 rows
train_ts = ts[:n_train]
test_ts = ts[n_train:]

# fixed hyperparameters
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

# rolling one-step-ahead forecasting
history = list(train_ts)
forecasts = []
for i in range(len(test_ts)):
    # fit SARIMA on current history
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=50, method='lbfgs')
    # forecast one step
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(pred)
    # update history with true value observed after prediction
    history.append(test_ts[i])

print(forecasts)