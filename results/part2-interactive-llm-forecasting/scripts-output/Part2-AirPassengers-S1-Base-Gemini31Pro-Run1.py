import random
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

train_size = 115
y_train = y[:train_size]

model = ExponentialSmoothing(y_train, trend='add', damped_trend=False, seasonal='mul', seasonal_periods=12)
fit_model = model.fit()

forecasts = fit_model.forecast(29).tolist()

print(forecasts)