import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df.set_index('Date', inplace=True)
series = df['Daily minimum temperatures'].astype(float).values
def replace_zeros(data):
    data = data.copy()
    for i in range(len(data)):
        if data[i] == 0:
            preceding = []
            j = i - 1
            while len(preceding) < 7 and j >= 0:
                if data[j] != 0:
                    preceding.append(data[j])
                j -= 1
            if preceding:
                data[i] = np.median(preceding)
            else:
                data[i] = np.mean(data[:i]) if i > 0 else 0
    return data
series = replace_zeros(series)
train_size = 2920
train = series[:train_size]
test = series[train_size:]
forecasts = []
current_train = train.copy()
for i in range(0, len(test), 30):
    h = min(30, len(test) - i)
    model = SARIMAX(current_train, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n', enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False, maxiter=100)
    pred = fitted_model.forecast(steps=h)
    forecasts.extend(pred.tolist())
    current_train = np.concatenate([current_train, test[i:i+h]])
print(forecasts)