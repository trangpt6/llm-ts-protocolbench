import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility seed
np.random.seed(42)

# read data
df = pd.read_csv(r'../../../data/AirPassengers.csv', index_col=0, parse_dates=True)
series = df['Passengers'].astype(float)

# chronological split
train_size = int(0.8 * len(series))  # 115
train = series.iloc[:train_size]
test = series.iloc[train_size:]

# fixed model hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'n'

forecasts = []

# expanding window rolling 1-step ahead forecast with retrain each step
for i in range(len(test)):
    # current training data: original train + all previous test points up to i-1
    if i == 0:
        current_train = train.copy()
    else:
        current_train = series.iloc[:train_size + i].copy()
    # fit SARIMA
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend)
    fitted = model.fit(disp=False)
    # forecast next step
    fc = fitted.forecast(steps=1)
    forecasts.append(fc.iloc[0] if hasattr(fc, 'iloc') else fc[0])

print(forecasts)