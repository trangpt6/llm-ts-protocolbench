import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
np.random.seed(42)

# Read the data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')

# Replace artificial constant-value blocks with NaN
artificial_dates = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31',
    '2018-03-31', '2018-05-31'
]
for d in artificial_dates:
    df.loc[d, 'OT'] = np.nan

# Linearly interpolate NaNs so that SARIMA can be fitted on complete data
df['OT'] = df['OT'].interpolate(method='linear')

# Chronological 80/20 split
train_size = 14016
train = df.iloc[:train_size]['OT']
test = df.iloc[train_size:]['OT']

# Initialise history with the entire training series
history = train.copy()

forecasts = []
model_order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)

# Rolling one-step ahead forecasting with retraining after each step
for i in range(len(test)):
    # Fit SARIMA on current history
    model = SARIMAX(history, order=model_order, seasonal_order=seasonal_order,
                    trend='c', enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False, maxiter=100)
    # Forecast the next hour
    fc = fit.forecast(steps=1).iloc[0]
    forecasts.append(fc)
    # Append the true observed value to history for the next iteration
    true_val = test.iloc[i]
    new_row = pd.Series([true_val], index=[test.index[i]])
    history = pd.concat([history, new_row])

print(forecasts)