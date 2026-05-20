import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# set random seed for reproducibility
np.random.seed(42)

# read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# parse dates and sort chronologically
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

# extract target series
y = df['Daily minimum temperatures']

# chronological 80/20 split
train_size = int(0.8 * len(y))
train = y.iloc[:train_size]
test = y.iloc[train_size:]

forecasts = []

# rolling multi-step forecast: retrain at every step with expanding window
for i in range(len(test)):
    history = y.iloc[:train_size + i]
    model = SARIMAX(history, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n')
    result = model.fit(disp=False)
    pred = result.forecast(steps=7)
    forecasts.append(float(pred.iloc[0]))

print(forecasts)