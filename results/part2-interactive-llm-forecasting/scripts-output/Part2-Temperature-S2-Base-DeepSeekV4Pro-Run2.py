import numpy as np
np.random.seed(42)

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)  # format M/D/YYYY
df = df.set_index("Date").sort_index()
y = df["Daily minimum temperatures"].astype(float)

# Train/test split: first int(0.8 * total) = 2921 train
train_len = int(0.8 * len(y))
train_series = y.iloc[:train_len]
test_series = y.iloc[train_len:]

forecasts = []

for i in range(len(test_series)):
    # data available up to the step before current prediction
    current_data = y.iloc[:train_len + i]  # indices 0 to train_len+i-1
    model = SARIMAX(
        current_data,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 365),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    # Fit with Nelder-Mead to handle long seasonal period robustly
    fit = model.fit(method='nm', maxiter=50, disp=0)
    pred = fit.forecast(steps=1)[0]
    forecasts.append(pred)

print(forecasts)