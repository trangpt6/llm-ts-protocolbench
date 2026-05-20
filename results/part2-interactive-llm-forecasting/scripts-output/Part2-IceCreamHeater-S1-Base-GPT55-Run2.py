import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").set_index("Month")
df = df.asfreq("MS")

y = df["Ice cream"]
train = y.iloc[:158]
test = y.iloc[158:198]

model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend="c"
)
result = model.fit(disp=False)

forecast_values = result.forecast(steps=len(test))
forecasts = [float(v) for v in forecast_values]

print(forecasts)