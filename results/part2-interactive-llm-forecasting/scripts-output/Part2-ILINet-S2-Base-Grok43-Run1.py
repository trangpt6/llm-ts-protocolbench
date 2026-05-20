import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# set seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values
train_size = 1039
train = target[:train_size].tolist()
test = target[train_size:]
forecasts = []
history = train.copy()
for t in range(len(test)):
    model = SARIMAX(history, order=(2, 0, 2), seasonal_order=(1, 1, 1, 52), trend='n')
    fitted = model.fit(disp=False)
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(pred)
    history.append(test[t])
print(forecasts)