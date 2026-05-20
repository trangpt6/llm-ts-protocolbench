import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
train = df['Passengers'].iloc[:115]
model = ExponentialSmoothing(train, damped_trend=False, seasonal='mul', seasonal_periods=12, trend='add')
fitted_model = model.fit()
forecasts = fitted_model.forecast(29).tolist()
print(forecasts)