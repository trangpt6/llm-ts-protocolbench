import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Ignore warnings to ensure only the forecast list is printed
warnings.filterwarnings("ignore")

# Read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
y = df['Daily minimum temperatures'].values

# Define train/test split strictly chronologically (80/20)
train_size = int(0.8 * len(y))
train = y[:train_size]
test = y[train_size:]

# Initialize history with training data
history = list(train)
forecasts = []

# Rolling update forecasting strategy
for i in range(len(test)):
    # Define and fit the SARIMA model with fixed hyperparameters
    model = sm.tsa.statespace.SARIMAX(
        history,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 365),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Retrain the model at every time step
    res = model.fit(disp=False)
    
    # Forecast 7 steps ahead
    pred = res.forecast(steps=7)
    
    # Append predictions to the flat list
    for p in pred:
        forecasts.append(float(p))
        
    # Update history with the true value for the next iteration
    history.append(test[i])

# Print only the final forecast list
print(forecasts)