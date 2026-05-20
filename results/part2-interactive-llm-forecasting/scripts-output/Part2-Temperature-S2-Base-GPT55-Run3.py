import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"])
target = df["Daily minimum temperatures"].astype(float).to_numpy()

train = target[:2920]
test = target[2920:3650]

history = [float(x) for x in train]
forecasts = []

for y_true in test:
    model = SARIMAX(
        history,
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, 365),
        trend="n"
    )
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=1)
    forecasts.append(float(np.asarray(pred)[0]))
    history.append(float(y_true))

print(forecasts)