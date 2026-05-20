import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
np.random.seed(42)

# Load the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
target_col = 'Daily minimum temperatures'

# Define the split point based on Turn 0
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train_data = df[target_col].iloc[:train_size].values

# Fixed hyperparameters from Turn 2
# order: [1, 0, 1], seasonal_order: [1, 1, 1, 365], trend: 'n'
order = (1, 0, 1)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

# Initialize and train the model once (Static setup)
model = SARIMAX(train_data, 
                order=order, 
                seasonal_order=seasonal_order, 
                trend=trend,
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit(disp=False)

# Forecast over the entire test set (730 steps)
# Ground truth during inference is disabled (no updates)
# Model retraining is not performed
forecasts = model_fit.forecast(steps=730).tolist()

# Print the final forecast list
print(forecasts)