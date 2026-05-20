import random
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility.
random.seed(0)
np.random.seed(0)
warnings.filterwarnings("ignore")

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").set_index("DATE")

target = "% WEIGHTED ILI"
y = df[target].astype(float).reset_index(drop=True)

train_size = 1040
test_size = 261
train = y.iloc[:train_size].copy()
test = y.iloc[train_size:train_size + test_size].copy()

order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = "n"

forecasts = []
for i in range(test_size):
    history = pd.concat([train, test.iloc[:i]], ignore_index=True)
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend
    )
    results = model.fit(disp=False)
    pred = results.forecast(steps=1)
    forecasts.append(float(np.asarray(pred)[0]))

print(forecasts)