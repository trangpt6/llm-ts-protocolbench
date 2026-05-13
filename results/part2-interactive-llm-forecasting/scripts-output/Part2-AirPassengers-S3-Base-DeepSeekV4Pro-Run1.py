import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Reproducibility seed
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
y = df['Passengers']

# Split
train_size = 115
n_test = len(y) - train_size

forecasts = []
for i in range(n_test):
    # Training data: all observations up to the point before the next test point
    train_data = y.iloc[:train_size + i]
    # Fit SARIMA(1,1,1)(1,1,1,12) with no constant
    model = ARIMA(train_data, order=(1,1,1), seasonal_order=(1,1,1,12), trend='n')
    fit = model.fit()
    # Forecast horizon 12, take only the immediate next step
    fcast = fit.forecast(steps=12)
    forecasts.append(fcast[0])

print(forecasts)