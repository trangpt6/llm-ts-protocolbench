import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values
forecasts = []
current_end = 115
while current_end < len(data):
    model = ExponentialSmoothing(data[:current_end], trend='add', damped_trend=False, seasonal='mul', seasonal_periods=12)
    fit = model.fit()
    remaining = len(data) - current_end
    h = min(12, remaining)
    pred = fit.forecast(h)
    forecasts.extend(pred.tolist())
    current_end += h
print(forecasts)