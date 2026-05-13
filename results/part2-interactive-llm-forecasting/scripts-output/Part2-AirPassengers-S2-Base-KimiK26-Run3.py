import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set random seed for reproducibility
np.random.seed(42)

# Read dataset from local CSV file
df = pd.read_csv(r'../../../data/AirPassengers.csv', index_col=0, parse_dates=True)
y = df['Passengers']

# Apply chronological 80/20 split
n_total = len(y)
train_size = int(0.8 * n_total)
train = y.iloc[:train_size]
test = y.iloc[train_size:]

forecasts = []

# Rolling one-step ahead forecast with retraining after each step
for i in range(len(test)):
    model = ExponentialSmoothing(
        train,
        trend='add',
        damped_trend=False,
        seasonal='mul',
        seasonal_periods=12
    )
    fitted = model.fit()
    fcast = fitted.forecast(steps=1)
    forecasts.append(float(fcast.iloc[0]))
    train = pd.concat([train, test.iloc[[i]]])

print(forecasts)