import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
# Resample to strict weekly frequency to handle missing weeks
df = df.resample('W-SUN').asfreq()
# Replace 0.0 with NaN and interpolate as per Turn 1
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].replace(0.0, np.nan)
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

target_col = '% WEIGHTED ILI'
series = df[target_col]

# Define split based on Turn 0
total_timesteps = len(series)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

train_data = series.iloc[:train_size].tolist()
test_data = series.iloc[train_size:].tolist()

# Fixed hyperparameters from Turn 2
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)

forecasts = []
history = list(train_data)

# Rolling update with retraining at every step
for i in range(test_size):
    model = SARIMAX(history, 
                    order=order, 
                    seasonal_order=seasonal_order, 
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    # One-step ahead forecast
    yhat = model_fit.forecast(steps=1)[0]
    forecasts.append(yhat)
    # Update history with ground truth for next step
    history.append(test_data[i])

print(forecasts)