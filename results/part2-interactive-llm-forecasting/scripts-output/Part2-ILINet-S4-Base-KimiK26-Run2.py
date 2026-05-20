import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# set random seed for reproducibility
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.set_index('DATE', inplace=True)

# target series
target = df['% WEIGHTED ILI']

# train/test split (first 1044 rows train, remaining test)
train = target.iloc[:1044]
test = target.iloc[1044:]
total_test_len = len(test)

# fixed model parameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

# block-wise rolling forecast
current_train = train.copy()
forecasts = []
position = 0

while position < total_test_len:
    horizon = min(52, total_test_len - position)
    # fit SARIMA model
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order,
                    trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200)
    # forecast next horizon steps
    pred = fitted.forecast(steps=horizon)
    forecasts.extend(pred.tolist())
    # get true values for this block from test set
    true_next = test.iloc[position:position+horizon]
    # update training set with true values (ground truth enabled)
    current_train = pd.concat([current_train, true_next])
    position += horizon

# output final forecast list
print(forecasts)