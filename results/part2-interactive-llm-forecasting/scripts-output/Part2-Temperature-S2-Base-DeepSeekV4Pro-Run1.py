import pandas as pd
import numpy as np
import warnings
from pmdarima import ARIMA

warnings.filterwarnings("ignore")
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], dayfirst=False)
df.sort_values('Date', inplace=True)
ts = df['Daily minimum temperatures'].astype(float).values

# Chronological split
train_size = int(0.8 * len(ts))  # 2921
train = ts[:train_size]
test = ts[train_size:]

forecasts = []
history = train.copy()

for true_val in test:
    model = ARIMA(order=(1,0,1), seasonal_order=(1,1,1,365), trend='n', suppress_warnings=True)
    # Fit on all available history
    model.fit(history)
    # One-step ahead forecast
    yhat = model.predict(n_periods=1)[0]
    forecasts.append(yhat)
    # Update history with observed true value
    history = np.append(history, true_val)

print(forecasts)