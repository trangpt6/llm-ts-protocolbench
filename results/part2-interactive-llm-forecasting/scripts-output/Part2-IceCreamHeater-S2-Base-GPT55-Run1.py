import random
random.seed(42)
import warnings
import numpy as np
np.random.seed(42)
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").set_index("Month")
df = df.asfreq("MS")

y = df["Ice cream"].astype(float)
train_size = 158
train = y.iloc[:train_size]
test = y.iloc[train_size:198]

history = train.copy()
forecasts = []

for true_value in test:
    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False)
    pred = result.forecast(steps=1).iloc[0]
    forecasts.append(float(pred))
    next_index = history.index[-1] + pd.offsets.MonthBegin(1)
    history = pd.concat([history, pd.Series([true_value], index=[next_index])])

print(forecasts)