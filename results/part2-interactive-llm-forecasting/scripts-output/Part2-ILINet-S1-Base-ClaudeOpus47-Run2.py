import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
df = df[["DATE", "% WEIGHTED ILI"]]
df = df.set_index("DATE")
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="7D")
df = df.reindex(full_idx)
df["% WEIGHTED ILI"] = df["% WEIGHTED ILI"].ffill()

n_total = 1301
n_train = 1040
n_test = 261

series = df["% WEIGHTED ILI"].iloc[:n_total]
train = series.iloc[:n_train]
test = series.iloc[n_train:n_train + n_test]

model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    trend="c",
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fitted = model.fit(disp=False)

fc = fitted.forecast(steps=n_test)
forecasts = [float(x) for x in np.asarray(fc).tolist()]

print(forecasts)