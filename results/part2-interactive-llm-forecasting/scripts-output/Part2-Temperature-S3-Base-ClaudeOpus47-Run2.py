import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)
df = df.set_index("Date")
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
df = df.reindex(full_idx)
df["Daily minimum temperatures"] = df["Daily minimum temperatures"].interpolate(method="linear")
y = df["Daily minimum temperatures"].astype(float).values

total = len(y)
train_size = int(0.8 * total)
test_size = total - train_size

H = 7
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 365)
trend = "n"

forecasts = []
for i in range(test_size):
    history = y[: train_size + i]
    try:
        model = SARIMAX(
            history,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=50, method="lbfgs")
        fc = res.forecast(steps=H)
        fc = np.asarray(fc, dtype=float)
    except Exception:
        fc = np.full(H, history[-1], dtype=float)
    remaining = test_size - i
    take = min(H, remaining)
    forecasts.extend(fc[:take].tolist())

forecasts = [float(v) for v in forecasts]
print(forecasts)