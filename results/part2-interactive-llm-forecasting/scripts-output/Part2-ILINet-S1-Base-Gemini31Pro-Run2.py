import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

df = df[['% WEIGHTED ILI']]
df = df.resample('W-SUN').asfreq()
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

model = SARIMAX(
    train_df['% WEIGHTED ILI'],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False)

forecast_steps = len(test_df)
forecasts = results.forecast(steps=forecast_steps)

print(forecasts.tolist())