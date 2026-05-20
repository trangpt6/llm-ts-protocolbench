import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = df['Ice cream'].values
train_size = 158
train_data = target[:train_size].tolist()
test_data = target[train_size:].tolist()
forecasts = []
current_train = train_data.copy()
i = 0
while i < len(test_data):
    model = SARIMAX(current_train, order=(1,1,1), seasonal_order=(1,1,1,12), trend='c')
    fitted_model = model.fit(disp=False)
    steps = min(12, len(test_data) - i)
    pred = fitted_model.forecast(steps=steps)
    forecasts.extend(pred.tolist())
    current_train.extend(test_data[i:i+steps])
    i += steps
print(forecasts)