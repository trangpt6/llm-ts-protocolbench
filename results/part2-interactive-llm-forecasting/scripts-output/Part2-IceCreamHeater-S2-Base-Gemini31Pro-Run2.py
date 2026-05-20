import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Set random seeds for reproducibility
np.random.seed(42)
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
data = df[target_col].values

# Define split based on Turn 0
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

train = list(data[:train_size])
test = data[train_size:]

# Fixed hyperparameters from Turn 2
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

forecasts = []

# Rolling update strategy: retrain at every step, one-step ahead forecast
for i in range(len(test)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, trend=trend)
    model_fit = model.fit(disp=False)
    
    # Forecast one step ahead
    pred = model_fit.forecast(steps=1)[0]
    forecasts.append(pred)
    
    # Update training set with ground truth (rolling update)
    train.append(test[i])

# Print final forecast list
print(forecasts)