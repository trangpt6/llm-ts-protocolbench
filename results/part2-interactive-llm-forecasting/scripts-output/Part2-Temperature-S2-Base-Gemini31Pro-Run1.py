import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Ignore warnings to keep output clean
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Extract target column
target = df['Daily minimum temperatures'].values

# Define train/test split
train_size = int(0.8 * len(target))
train = target[:train_size]
test = target[train_size:]

# Initialize history with training data
history = list(train)
forecasts = []

# Rolling forecast with retraining at every step
for t in range(len(test)):
# Define and fit the model
    model = SARIMAX(history, order=(1, 0, 1), seasonal_order=(1, 1, 1, 365), trend='n')
    model_fit = model.fit(disp=False)
    
# Forecast one step ahead
    yhat = model_fit.forecast()[0]
    forecasts.append(yhat)
    
# Add ground truth to history for the next step
    history.append(test[t])

# Print final forecast list
print(forecasts)