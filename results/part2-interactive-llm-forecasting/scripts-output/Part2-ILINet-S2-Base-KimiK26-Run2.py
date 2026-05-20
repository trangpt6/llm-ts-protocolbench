import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# Convert DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Sort chronologically
df = df.sort_values('DATE').reset_index(drop=True)

# Preprocessing: create binary indicator for zero target periods
df['zero_indicator'] = (df['% WEIGHTED ILI'] == 0).astype(int)

# Preprocessing: forward-fill missing age category values within valid periods
for col in ['AGE 25-49', 'AGE 25-64', 'AGE 50-64']:
    if col in df.columns:
        df[col] = df[col].ffill()

# Define target column
target_col = '% WEIGHTED ILI'

# Chronological 80/20 split
n = len(df)
train_size = int(0.8 * n)
train_series = df[target_col].iloc[:train_size].reset_index(drop=True)
test_values = df[target_col].iloc[train_size:].reset_index(drop=True)

# Fixed hyperparameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

forecasts = []

# Rolling one-step ahead forecasting with retraining
for i in range(len(test_values)):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    fcast = fitted.forecast(steps=1)
    val = float(fcast.iloc[0])
    forecasts.append(val)
    # Append observed true value for next iteration
    train_series = pd.concat([train_series, pd.Series([test_values.iloc[i]])], ignore_index=True)

print(forecasts)