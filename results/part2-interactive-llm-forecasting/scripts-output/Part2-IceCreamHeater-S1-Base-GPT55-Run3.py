import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").set_index("Month")
df = df.asfreq("MS")

train = df.iloc[:158]
test = df.iloc[158:198]

model = SARIMAX(
    train["Ice cream"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    trend="c"
)
result = model.fit(disp=False)
forecast_values = result.forecast(steps=40)
forecasts = [float(x) for x in forecast_values.tolist()]
print(forecasts)