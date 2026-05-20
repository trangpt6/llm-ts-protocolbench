import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

y = df["Ice cream"].astype(float)

train_size = 158
test_size = 40
train_y = y.iloc[:train_size]
test_y = y.iloc[train_size:train_size + test_size]

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = "c"
horizon = 12

forecasts = []
for i in range(test_size):
    history = y.iloc[:train_size + i]
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=horizon)
    valid_steps = min(horizon, test_size - i)
    forecasts.extend([float(v) for v in pred.iloc[:valid_steps]])

print(forecasts)