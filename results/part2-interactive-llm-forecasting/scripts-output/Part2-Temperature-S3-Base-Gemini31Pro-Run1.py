import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Preprocessing steps
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Define target and split
target_col = 'Daily minimum temperatures'
data = df[target_col].values
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
train_data = data[:train_size]
test_data = data[train_size:]

# Initialize history and forecasts
history = list(train_data)
forecasts = []
horizon = 7

# Rolling update loop
for i in range(len(test_data)):
    # Initialize and train model
    model = SARIMAX(history, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend='n')
    fit_model = model.fit(disp=False)
    
    # Forecast multiple steps
    yhat = fit_model.forecast(steps=horizon)
    forecasts.extend(yhat.tolist())
    
    # Update history with ground truth
    history.append(test_data[i])

# Print final forecast list
print(forecasts)