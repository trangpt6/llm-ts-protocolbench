import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore")

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").set_index("Month")
df.index = pd.PeriodIndex(df.index, freq="M").to_timestamp()

y = df["Ice cream"].astype(float)

train_size = 158
test_size = 40
train_y = y.iloc[:train_size]
test_y = y.iloc[train_size:train_size + test_size]

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = "c"
block_size = 12
horizon = 12

forecasts = []
history = train_y.copy()

for start in range(0, test_size, block_size):
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend)
    fit = model.fit(disp=False)
    pred = fit.forecast(steps=horizon)
    remaining = test_size - start
    take = min(block_size, remaining)
    forecasts.extend([float(x) for x in pred.iloc[:take]])
    true_block = test_y.iloc[start:start + take]
    history = pd.concat([history, true_block])

print(forecasts)