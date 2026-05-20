import pandas as pd
import numpy as np
import statsmodels.api as sm
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Read the CSV file
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing: drop entirely empty columns
df.drop(columns=['AGE 25-49', 'AGE 25-64'], inplace=True, errors='ignore')

# Ensure date column is datetime and sort chronologically
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)
df.reset_index(drop=True, inplace=True)

# Define target series
target_col = '% WEIGHTED ILI'
series = df[target_col].values.astype(float)

# Chronological split: 80% train, 20% test
train_size = int(0.8 * len(series))
train_series = list(series[:train_size])
test_series = series[train_size:].tolist()

# Fixed model and hyperparameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

# Rolling one-step ahead forecasting with retraining
forecasts = []
current_train = train_series[:]

for i, true_val in enumerate(test_series):
    # Fit SARIMA on current training series
    model = sm.tsa.SARIMAX(current_train,
                           order=order,
                           seasonal_order=seasonal_order,
                           trend=trend,
                           enforce_stationarity=False,
                           enforce_invertibility=False)
    fitted = model.fit(disp=False, maxiter=100)
    # Forecast one step
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(float(pred))
    # Update training set with true value (use of ground truth)
    current_train.append(true_val)

# Output the final forecast list
print(forecasts)