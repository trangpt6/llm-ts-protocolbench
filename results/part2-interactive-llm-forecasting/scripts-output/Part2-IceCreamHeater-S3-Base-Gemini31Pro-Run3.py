import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Extract target
y = df['Ice cream'].values

# Train/test split
total_timesteps = len(y)
train_size = int(0.8 * total_timesteps)
train_data = y[:train_size]
test_data = y[train_size:]

# Forecasting setup
horizon = 12
forecasts = []

# Rolling update with retraining
history = list(train_data)

for i in range(len(test_data)):
    # Define and fit the model
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    model_fit = model.fit(disp=False)
    
    # Forecast 12 steps ahead
    yhat = model_fit.forecast(steps=horizon)
    forecasts.extend(yhat.tolist())
    
    # Update history with the true value for the next step
    history.append(test_data[i])

# Print the final flat list of forecasts
print(forecasts)