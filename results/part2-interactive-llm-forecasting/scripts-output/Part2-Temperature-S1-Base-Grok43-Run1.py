import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].astype(float).values
train_data = target[:2920]
model = SARIMAX(train_data, order=(1, 0, 1), seasonal_order=(1, 1, 1, 365), trend='n')
fitted_model = model.fit(disp=False)
forecasts = fitted_model.forecast(steps=730).tolist()
print(forecasts)