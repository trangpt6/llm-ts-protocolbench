import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

train_size = 115
history = df['Passengers'].iloc[:train_size].astype(float).tolist()
test_values = df['Passengers'].iloc[train_size:].astype(float).tolist()

forecasts = []
for true_val in test_values:
    series = pd.Series(history, dtype=float)
    model = ExponentialSmoothing(
        series,
        trend='add',
        seasonal='mul',
        seasonal_periods=12,
        damped_trend=False
    )
    fitted = model.fit()
    pred = fitted.forecast(1).iloc[0]
    forecasts.append(pred)
    history.append(true_val)

print(forecasts)