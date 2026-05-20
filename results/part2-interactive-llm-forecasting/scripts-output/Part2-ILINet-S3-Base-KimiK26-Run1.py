import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# reproducibility
np.random.seed(42)
warnings.filterwarnings("ignore")

# load dataset
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.set_index('DATE', inplace=True)

# reindex to complete weekly Sunday frequency to expose implicit missing dates
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_idx)

# mark structural reporting gaps as missing in target where providers and patients are zero
gap_mask = (df['NUM. OF PROVIDERS'] == 0) & (df['TOTAL PATIENTS'] == 0)
df.loc[gap_mask, '% WEIGHTED ILI'] = np.nan

# target series
y = df['% WEIGHTED ILI']

# fixed chronological split boundary from Turn 0
split_date = pd.Timestamp('2017-10-01')
train_size = y.index.get_loc(split_date) + 1
test_size = len(y) - train_size
horizon = 4

forecasts = []
for i in range(test_size):
    train_y = y.iloc[:train_size + i]
    model = SARIMAX(train_y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c')
    results = model.fit(disp=False)
    fcast = results.forecast(steps=horizon)
    forecasts.extend(fcast.tolist())

print(forecasts)