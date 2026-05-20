import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set seed for reproducibility.
np.random.seed(0)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29

train = y[:train_size]
test = y[train_size:train_size + test_size]

history = list(train)
forecasts = []

for i in range(len(test)):
    model = ExponentialSmoothing(
        history,
        damped_trend=False,
        seasonal="mul",
        seasonal_periods=12,
        trend="add"
    )
    fitted = model.fit()
    pred = fitted.forecast(1)[0]
    forecasts.append(float(pred))
    history.append(float(test[i]))

print(forecasts)