import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

target_col = "% WEIGHTED ILI"
y = df[target_col].astype(float).to_numpy()

train_size = 1040
test_size = 261
train_y = y[:train_size]

model = SARIMAX(
    train_y,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    trend="c"
)
result = model.fit(disp=False)

forecasts = result.forecast(steps=test_size)
forecasts = [float(x) for x in np.asarray(forecasts).ravel().tolist()]

print(forecasts)