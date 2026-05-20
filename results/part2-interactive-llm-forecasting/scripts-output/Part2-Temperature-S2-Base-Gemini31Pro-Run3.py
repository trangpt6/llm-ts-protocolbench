import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

# Set random seed for reproducibility
np.random.seed(42)

# Ignore warnings to ensure only the forecast list is printed
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Apply preprocessing steps
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index.freq = 'D'

# Define train and test sets
train_size = int(0.8 * len(df))
train = df.iloc[:train_size]['Daily minimum temperatures']
test = df.iloc[train_size:]['Daily minimum temperatures']

# Initialize history with training data
history = list(train.values)
forecasts = []

# Rolling forecast with retraining at every step
for i in range(len(test)):
    # Initialize and fit SARIMA model
    model = sm.tsa.SARIMAX(history, order=(1, 0, 1), seasonal_order=(1, 1, 1, 365), trend='n')
    fitted_model = model.fit(disp=False)
    
    # Predict one step ahead
    yhat = fitted_model.forecast(steps=1)[0]
    forecasts.append(yhat)
    
    # Append true value to history for the next iteration
    history.append(test.iloc[i])

# Print final forecast list
print(forecasts)