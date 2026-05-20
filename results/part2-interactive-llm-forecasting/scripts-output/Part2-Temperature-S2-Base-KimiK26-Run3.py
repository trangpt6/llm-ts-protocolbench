import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read and prepare data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
df = df.sort_values('Date').reset_index(drop=True)
y = df['Daily minimum temperatures'].astype(float).values

# chronological split
train_size = int(0.8 * len(y))
train_data = y[:train_size]
test_data = y[train_size:]

# fixed hyperparameters
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

forecasts = []
history = train_data.tolist()

# rolling one-step ahead forecasting with retraining after each step
for true_val in test_data:
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend,
                    enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(float(pred))
    history.append(float(true_val))

print(forecasts)