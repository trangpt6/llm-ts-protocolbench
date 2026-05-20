import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

warnings.filterwarnings("ignore")

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

target_col = "Daily minimum temperatures"
y = df[target_col].astype(float).to_numpy()

train_size = 2920
test_size = 730
train = y[:train_size]
test = y[train_size:train_size + test_size]

order = (2, 0, 2)
seasonal_order = (1, 1, 1, 365)
trend = "n"
block_size = 30
horizon = 30

history = list(train)
forecasts = []

for start in range(0, test_size, block_size):
    steps = min(horizon, test_size - start)
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend)
    fitted = model.fit(disp=False)
    block_forecast = fitted.forecast(steps=steps)
    forecasts.extend([float(v) for v in block_forecast])
    history.extend([float(v) for v in test[start:start + steps]])

forecasts = forecasts[:test_size]
print(forecasts)