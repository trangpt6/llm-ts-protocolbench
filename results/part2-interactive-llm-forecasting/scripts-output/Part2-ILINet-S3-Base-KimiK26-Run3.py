import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)
y = df['% WEIGHTED ILI'].values.astype(float)

# Split: first 1043 for training, rest for test
train_size = 1043
train = y[:train_size]
test = y[train_size:]
test_len = len(test)

# Fixed hyperparameters
order = [1, 1, 1]
seasonal_order = [1, 1, 1, 52]
trend = 'c'

# Rolling forecast with horizon 4, window advancement 1
forecasts = []
history = list(train)

for i in range(test_len):
    # Fit SARIMA on current history
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=200, low_memory=True)
    # Forecast next 4 steps, take first step
    pred = fitted.forecast(steps=4)[0]
    forecasts.append(pred)
    # Update history with true value (ground truth)
    history.append(test[i])

print(forecasts)