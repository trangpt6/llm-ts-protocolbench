import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Set random seed
np.random.seed(42)

# Read data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

# Split data
train_size = 115
test_size = len(y) - train_size

forecasts = []

# Rolling update with retraining
for i in range(test_size):
    train_data = y[:train_size + i]
    model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='n')
    fit_model = model.fit(disp=False)
    pred = fit_model.forecast(steps=12)
    forecasts.extend(pred.tolist())

print(forecasts)