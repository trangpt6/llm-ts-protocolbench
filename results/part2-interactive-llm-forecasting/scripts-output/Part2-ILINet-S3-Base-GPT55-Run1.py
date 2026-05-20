import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set seed for reproducibility.
np.random.seed(0)

df = pd.read_csv(r'../../../data/ILINet.csv')
df = df[["DATE", "% WEIGHTED ILI"]].copy()
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

target = df["% WEIGHTED ILI"].astype(float).reset_index(drop=True)

train_size = 1040
test_size = 261
horizon = 4

train = target.iloc[:train_size]
test = target.iloc[train_size:train_size + test_size]

forecasts = []

for i in range(test_size):
    history = target.iloc[:train_size + i]
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        trend="c"
    )
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=horizon)
    steps_to_keep = min(horizon, test_size - i)
    forecasts.extend([float(x) for x in pred.iloc[:steps_to_keep]])

print(forecasts)