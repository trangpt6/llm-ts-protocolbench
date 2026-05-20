import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Read dataset from local CSV
df = pd.read_csv(r'../../../data/Temperature.csv')

# Parse time index and sort chronologically
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Extract primary target
y = df['Daily minimum temperatures'].astype(float).values

# Apply chronological 80/20 split
train_size = 2920
train = y[:train_size].tolist()
test = y[train_size:].tolist()

# Fixed SARIMA hyperparameters
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

# Rolling one-step ahead forecast with retraining after each step
forecasts = []
history = train[:]

for i in range(len(test)):
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    fc = fitted.forecast(steps=1)
    forecasts.append(float(fc[0]))
    history.append(test[i])

print(forecasts)