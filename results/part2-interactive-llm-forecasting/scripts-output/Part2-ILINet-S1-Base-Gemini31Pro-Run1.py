import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
import warnings

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

df = df.resample('W-SUN').asfreq()
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].replace(0.0, np.nan)
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train = df['% WEIGHTED ILI'].iloc[:train_size]
test_size = total_timesteps - train_size

model = sm.tsa.SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c')
results = model.fit(disp=False)

forecasts = results.forecast(steps=test_size)
print(forecasts.tolist())