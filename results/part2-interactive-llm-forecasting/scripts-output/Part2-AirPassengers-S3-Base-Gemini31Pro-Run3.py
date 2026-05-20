import pandas as pd
import numpy as np
import random
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

warnings.filterwarnings("ignore")

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values

train_size = 115
test_size = len(y) - train_size

forecasts = []

for i in range(test_size):
    current_train = y[:train_size + i]
    model = SARIMAX(current_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='n')
    fit_model = model.fit(disp=False)
    pred = fit_model.forecast(steps=12)
    forecasts.extend(pred.tolist())

print(forecasts)