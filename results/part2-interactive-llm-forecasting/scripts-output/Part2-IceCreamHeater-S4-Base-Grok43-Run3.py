import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
target = df['Ice cream']
train = target.iloc[:158]
test = target.iloc[158:]
forecasts = []
current_train = train.copy()
i = 0
test_values = test.values
while i < len(test):
    model = SARIMAX(current_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c', enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)
    steps = min(12, len(test) - i)
    pred = fitted_model.forecast(steps=steps)
    forecasts.extend(pred.tolist())
    true_vals = test_values[i:i+steps]
    last_date = current_train.index[-1]
    new_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
    true_series = pd.Series(true_vals, index=new_dates)
    current_train = pd.concat([current_train, true_series])
    i += steps
print(forecasts)