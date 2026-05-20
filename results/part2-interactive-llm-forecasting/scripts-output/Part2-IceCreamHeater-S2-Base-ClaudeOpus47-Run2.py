import warnings
warnings.filterwarnings("ignore")
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)
df = df.set_index("Month").asfreq("MS")

y = df["Ice cream"].astype(float)

total = len(y)
train_size = int(0.8 * total)
test_size = total - train_size

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = "c"

forecasts = []
history = y_train.copy()

for i in range(test_size):
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.forecast(steps=1)
    yhat = float(fc.iloc[0])
    forecasts.append(yhat)
    next_idx = y_test.index[i]
    history = pd.concat([history, pd.Series([y_test.iloc[i]], index=[next_idx])])
    history = history.asfreq("MS")

print(forecasts)