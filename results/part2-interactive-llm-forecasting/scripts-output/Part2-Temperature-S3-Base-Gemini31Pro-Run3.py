import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)
warnings.filterwarnings("ignore")

# Read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Apply preprocessing steps
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)

# Define target and split
target = df["Daily minimum temperatures"].values
train_size = 2921
train_target = target[:train_size]
test_target = target[train_size:]

# Initialize history with training data
history = list(train_target)
forecasts = []

# Rolling update forecasting
for i in range(len(test_target)):
    # Initialize and train the model
    model = SARIMAX(history, order=(2, 0, 2), seasonal_order=(1, 1, 1, 365), trend="n")
    fit_model = model.fit(disp=False)
    
    # Forecast 7 steps ahead
    yhat = fit_model.forecast(steps=7)
    forecasts.extend(yhat.tolist())
    
    # Update history with the true value for the next iteration
    history.append(test_target[i])

# Print final forecast list
print(forecasts)