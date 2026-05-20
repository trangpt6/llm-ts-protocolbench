import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].tolist()
train_size = 115
train = data[:train_size]
test = data[train_size:]
forecasts = []
current_train = train[:]
i = 0
while i < len(test):
    model = ExponentialSmoothing(current_train, trend='add', damped_trend=False, seasonal='mul', seasonal_periods=12)
    fit = model.fit()
    horizon = min(12, len(test) - i)
    pred = fit.forecast(horizon)
    forecasts.extend(pred.tolist())
    current_train.extend(test[i:i+horizon])
    i += horizon
print(forecasts)