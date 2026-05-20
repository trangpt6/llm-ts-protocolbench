import pandas as pd
import numpy as np
import statsmodels.api as sm
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')

# Convert Month to datetime and set as index
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').asfreq('MS')  # Monthly start frequency
target = df['Ice cream']

# No preprocessing needed as per Turn 1

# Chronological split: first 158 train, rest test
train_size = 158
train = target.iloc[:train_size]
test = target.iloc[train_size:]

# Fixed SARIMA hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

# Rolling one-step ahead forecast with retraining
forecasts = []
current_train = train.copy()

for true_val in test:
    # Fit SARIMA with fixed parameters
    model = sm.tsa.SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend)
    fitted = model.fit(disp=False, maxiter=200, method='lbfgs')
    
    # Forecast one step ahead
    pred = fitted.forecast(steps=1)
    forecasts.append(pred.iloc[0])
    
    # Update training data with actual ground truth
    current_train = pd.concat([current_train, pd.Series([true_val], index=[test.index[len(forecasts)-1]])])

# Output final forecast list
print(forecasts)