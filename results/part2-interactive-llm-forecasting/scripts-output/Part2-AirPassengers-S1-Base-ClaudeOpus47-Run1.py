import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

n = len(df)
train_size = int(0.8 * n)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

y_train = train['Passengers'].astype(float).values
h = len(test)

model = ExponentialSmoothing(
    y_train,
    trend='add',
    seasonal='mul',
    seasonal_periods=12,
    damped_trend=False,
)
fit = model.fit()
forecasts = [float(v) for v in fit.forecast(h)]

print(forecasts)