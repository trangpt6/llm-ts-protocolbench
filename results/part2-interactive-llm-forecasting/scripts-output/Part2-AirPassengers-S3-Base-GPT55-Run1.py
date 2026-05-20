import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set seeds for reproducibility.
random.seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore")

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

y = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
train = y[:train_size]
test = y[train_size:train_size + test_size]

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = "n"
horizon = 12

history = list(train)
forecasts = []

for i in range(test_size):
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend
    )
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=horizon)
    forecasts.append(float(pred[0]))
    history.append(float(test[i]))

print(forecasts)