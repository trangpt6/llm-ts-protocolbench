import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seeds for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = df['Ice cream'].values
train_size = 158
current_train = target[:train_size].tolist()
test = target[train_size:].tolist()
forecasts = []
for i in range(len(test)):
    model = SARIMAX(current_train, order=[1,1,1], seasonal_order=[1,1,1,12], trend='c')
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=12)[0]
    forecasts.append(pred)
    current_train.append(test[i])
print(forecasts)