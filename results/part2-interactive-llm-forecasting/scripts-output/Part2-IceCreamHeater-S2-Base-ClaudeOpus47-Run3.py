import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)
df = df.set_index("Month")

y = df["Ice cream"].astype(float)

n = len(y)
train_size = int(0.8 * n)
train = y.iloc[:train_size]
test = y.iloc[train_size:]

history = list(train.values)
forecasts = []

for i in range(len(test)):
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    yhat = res.forecast(steps=1)
    yhat_val = float(np.asarray(yhat)[0])
    forecasts.append(yhat_val)
    history.append(float(test.iloc[i]))

print(forecasts)