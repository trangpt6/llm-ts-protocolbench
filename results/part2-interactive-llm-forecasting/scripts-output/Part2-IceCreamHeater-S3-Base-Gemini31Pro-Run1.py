import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Ignore warnings
warnings.filterwarnings("ignore")

# Read dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# Preprocessing
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.index.freq = 'MS'

# Train test split
y = df['Ice cream'].values
train_size = int(0.8 * len(y))
train = y[:train_size]
test = y[train_size:]

# Forecasting setup
forecasts = []
history = list(train)

# Rolling update with retraining
for i in range(len(test)):
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    model_fit = model.fit(disp=False)
    yhat = model_fit.forecast(steps=12)
    forecasts.extend(yhat.tolist())
    history.append(test[i])

print(forecasts)