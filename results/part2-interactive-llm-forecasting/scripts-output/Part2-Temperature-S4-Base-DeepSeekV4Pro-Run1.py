import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds
np.random.seed(42)

# Load and preprocess
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.set_index('Date')
df = df.reindex(pd.date_range(start='1981-01-01', end='1990-12-31', freq='D'))
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='linear')

# Chronological split
train_size = int(0.8 * len(df))
train = df['Daily minimum temperatures'].iloc[:train_size]
test = df['Daily minimum temperatures'].iloc[train_size:]

# Block-wise rolling forecast
block_size = 30
current_train = train.copy()
forecasts = []

for i in range(0, len(test), block_size):
    remaining = len(test) - i
    steps = min(block_size, remaining)
    
    model = SARIMAX(current_train,
                    order=(2, 0, 2),
                    seasonal_order=(1, 1, 1, 365),
                    trend='n')
    model_fit = model.fit(disp=False)
    
    pred = model_fit.forecast(steps=steps)
    forecasts.extend(pred.tolist())
    
    current_train = pd.concat([current_train, test.iloc[i:i+steps]])

print(forecasts)