import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = df['Ice cream'].values

# Define split point based on Turn 0
train_size = 158
test_size = 40

# Fixed hyperparameters from Turn 2
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

forecasts = []
history = list(target[:train_size])

# Rolling update strategy: retrain and predict one-step ahead
for i in range(test_size):
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend)
    model_fit = model.fit(disp=False)
    
    # Forecast one step ahead
    yhat = model_fit.forecast(steps=1)[0]
    forecasts.append(yhat)
    
    # Update history with ground truth for next iteration
    actual = target[train_size + i]
    history.append(actual)

print(forecasts)