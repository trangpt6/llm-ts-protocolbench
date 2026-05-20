import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read the dataset
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')
df = df.asfreq('h')

# Define split
total = len(df)
train_size = int(0.8 * total)
train = df['OT'].iloc[:train_size]
test = df['OT'].iloc[train_size:]

# Define and fit SARIMA model
model = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24), trend='c')
fit = model.fit(disp=False)

# Generate dynamic forecasts for the entire test period
forecasts = fit.predict(start=train_size, end=total - 1, dynamic=True)

# Convert to list
forecasts = forecasts.tolist()

print(forecasts)