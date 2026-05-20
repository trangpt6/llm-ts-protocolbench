import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").set_index("Month")

y = df["Ice cream"].astype(float)

train_size = 158
test_size = 40
train = y.iloc[:train_size]
test = y.iloc[train_size:train_size + test_size]

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = "c"
horizon = 12

forecasts = []
for i in range(test_size):
    history = pd.concat([train, test.iloc[:i]])
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend
    )
    fitted = model.fit(disp=False)
    steps = min(horizon, test_size - i)
    pred = fitted.forecast(steps=steps)
    forecasts.extend([float(v) for v in pred.iloc[:steps]])

print(forecasts)