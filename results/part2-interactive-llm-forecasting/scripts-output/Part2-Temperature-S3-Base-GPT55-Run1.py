import random
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Reproducibility
random.seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore")

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df["Daily minimum temperatures"] = pd.to_numeric(df["Daily minimum temperatures"])
df = df.sort_values("Date").reset_index(drop=True)

train_raw = df.iloc[:2920].copy()
test_raw = df.iloc[2920:3650].copy()

target = "Daily minimum temperatures"
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 365)
trend = "n"
horizon = 7

history = train_raw.set_index("Date")[target].copy()
test_values = test_raw[target].to_numpy()
test_dates = test_raw["Date"].to_numpy()

forecasts = [np.nan] * 730

for i in range(730):
    regular_index = pd.date_range(history.index.min(), history.index.max(), freq="D")
    y_train = history.reindex(regular_index)
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend
    )
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=horizon)
    pred_values = np.asarray(pred, dtype=float)
    for k in range(horizon):
        j = i + k
        if j < 730 and np.isnan(forecasts[j]):
            forecasts[j] = float(pred_values[k])
    history.loc[pd.Timestamp(test_dates[i])] = float(test_values[i])
    history = history.sort_index()

print(forecasts)