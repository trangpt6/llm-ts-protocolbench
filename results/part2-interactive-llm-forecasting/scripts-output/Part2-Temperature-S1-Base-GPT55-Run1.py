import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

target_col = "Daily minimum temperatures"
y = df[target_col].astype(float)

train = y.iloc[:2920]
test = y.iloc[2920:3650]

model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 1, 1, 365),
    trend="n"
)

result = model.fit(disp=False)
forecast_result = result.get_forecast(steps=len(test))
forecasts = np.asarray(forecast_result.predicted_mean, dtype=float).tolist()

print(forecasts)