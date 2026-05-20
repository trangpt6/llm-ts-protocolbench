import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Ignore warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
y = df['Ice cream'].values

# Train/test split
train_size = int(0.8 * len(y))
train = list(y[:train_size])
test = list(y[train_size:])

# Forecasting setup
forecasts = []
horizon = 12

# Rolling update with retraining
for i in range(len(test)):
    current_train = train + test[:i]
    model = sm.tsa.SARIMAX(current_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    res = model.fit(disp=False)
    fc = res.forecast(steps=horizon)
    forecasts.extend(fc.tolist())

print(forecasts)