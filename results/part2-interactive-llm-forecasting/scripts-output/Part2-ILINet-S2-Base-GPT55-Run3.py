import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Reproducibility seed setting
random.seed(0)
np.random.seed(0)

df = pd.read_csv(r'../../../data/ILINet.csv')
series = df[["DATE", "% WEIGHTED ILI"]].copy()
series["DATE"] = pd.to_datetime(series["DATE"])
y = series["% WEIGHTED ILI"].astype(float).reset_index(drop=True)

train_size = 1040
test_size = 261

train = y.iloc[:train_size].copy()
test = y.iloc[train_size:train_size + test_size].copy()

history = train.tolist()
forecasts = []

order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = "n"

for i in range(len(test)):
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend
    )
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(float(pred))
    history.append(float(test.iloc[i]))

print(forecasts)