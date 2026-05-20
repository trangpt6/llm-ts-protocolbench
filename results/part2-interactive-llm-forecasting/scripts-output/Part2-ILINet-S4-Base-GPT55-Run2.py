import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/ILINet.csv')
df = df[["DATE", "% WEIGHTED ILI"]].copy()
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

y = df["% WEIGHTED ILI"].astype(float)

train_size = 1040
test_size = 261
train = y.iloc[:train_size].copy()
test = y.iloc[train_size:train_size + test_size].copy()

order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = "n"
block_size = 52
horizon = 52

history = train.copy()
forecasts = []
start = 0

while start < len(test):
    steps = min(block_size, horizon, len(test) - start)
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend
    )
    results = model.fit(disp=False)
    pred = results.forecast(steps=steps)
    forecasts.extend(np.asarray(pred, dtype=float).tolist())
    history = pd.concat([history, test.iloc[start:start + steps]], ignore_index=True)
    start += steps

forecasts = forecasts[:test_size]
print(forecasts)