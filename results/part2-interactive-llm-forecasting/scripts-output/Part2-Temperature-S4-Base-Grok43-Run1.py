import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df.iloc[:, 1].values
train_size = 2920
train = target[:train_size].tolist()
test = target[train_size:].tolist()
forecasts = []
current_train = train.copy()
i = 0
block_size = 30
while i < len(test):
    model = SARIMAX(current_train, order=[2, 0, 2], seasonal_order=[1, 1, 1, 365], trend='n')
    fitted_model = model.fit(disp=False)
    remaining = len(test) - i
    h = min(block_size, remaining)
    pred = fitted_model.forecast(steps=h)
    forecasts.extend(pred.tolist())
    current_train.extend(test[i:i+h])
    i += h
print(forecasts)