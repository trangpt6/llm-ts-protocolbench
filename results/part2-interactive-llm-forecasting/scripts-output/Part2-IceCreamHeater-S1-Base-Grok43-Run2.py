import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = df['Ice cream']
train = target.iloc[:158]
model = SARIMAX(train, order=[1, 1, 1], seasonal_order=[1, 1, 1, 12], trend='c')
fitted_model = model.fit(disp=False)
forecasts = fitted_model.forecast(steps=40).tolist()
print(forecasts)