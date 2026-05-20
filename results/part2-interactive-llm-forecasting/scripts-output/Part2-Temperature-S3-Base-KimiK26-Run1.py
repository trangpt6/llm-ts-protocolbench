import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(0)

# Read dataset from local CSV
df = pd.read_csv(r'../../../data/Temperature.csv')

# Parse time index and target
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').set_index('Date')
y = df['Daily minimum temperatures'].astype(float)
y.index.freq = 'D'

# Chronological 80/20 split
n_total = len(y)
n_train = int(0.8 * n_total)
train_data = y.iloc[:n_train].copy()
test_data = y.iloc[n_train:].copy()

forecasts = []

# Rolling multi-step forecast: horizon 7, advance 1, retrain each step, use ground truth
for i in range(len(test_data)):
    model = SARIMAX(train_data, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n')
    result = model.fit(disp=False)
    
    # Predict 7 steps ahead and retain the first step for the current test point
    pred = result.get_forecast(steps=7).predicted_mean.iloc[0]
    forecasts.append(float(pred))
    
    # Update training data with observed true value
    train_data = pd.concat([train_data, test_data.iloc[[i]]])
    train_data.index.freq = 'D'

print(forecasts)