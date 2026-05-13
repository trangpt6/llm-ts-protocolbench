import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Reproducibility
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
df.index.freq = 'MS'

# Chronological split
total = len(df)
train_size = int(0.8 * total)  # 115
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Fixed hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'n'

# Rolling forecast
history = train['Passengers'].copy()
forecasts = []

for i in range(len(test)):
    # Fit model on current history
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend)
    model_fit = model.fit(disp=False)
    # Forecast next 12 steps
    pred = model_fit.forecast(steps=12)
    # The one-step-ahead forecast is the first value (index 0)
    forecast_next = pred.iloc[0]
    forecasts.append(forecast_next)
    # Append true value to history
    true_val = test['Passengers'].iloc[i]
    history = pd.concat([history, pd.Series([true_val], index=[test.index[i]])])

print(forecasts)