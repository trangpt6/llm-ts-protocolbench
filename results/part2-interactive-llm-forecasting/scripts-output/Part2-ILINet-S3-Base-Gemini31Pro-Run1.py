import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Set random seeds for reproducibility
np.random.seed(42)
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE')

# Insert missing timestamp 2002-01-06 and interpolate
df = df.resample('W-SUN').asfreq()
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# Fill missing exogenous columns with 0
exog_cols = ['AGE 25-49', 'AGE 50-64']
for col in exog_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Define split point
train_size = 1044
test_size = 261
target_col = '% WEIGHTED ILI'

# Forecasting setup: Rolling update, retrain every step, horizon H=4
# To cover the test set (indices 1044 to 1304) with a 4-step horizon,
# we start predicting at index 1040 (to get index 1044) up to 1300 (to get index 1304).
forecasts = []
h = 4

# The test set starts at index 1044. To predict index 1044 at H=4, we must be at index 1040.
# We iterate through the test period, retraining the model at each step.
for i in range(test_size):
    # Current time index in the rolling window
    current_idx = (train_size - h) + i
    
    # Training data: all data up to current_idx (inclusive)
    train_series = df[target_col].iloc[:current_idx + 1]
    
    # SARIMA model with fixed hyperparameters
    # order: [1, 1, 1], seasonal_order: [1, 1, 1, 52], trend: 'c'
    model = SARIMAX(
        train_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52),
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    try:
        res = model.fit(disp=False)
        # Forecast H steps ahead
        pred = res.forecast(steps=h)
        # Take the H-th value to form the flat list covering the test set
        forecasts.append(float(pred.iloc[-1]))
    except:
        # Fallback in case of convergence issues
        if len(forecasts) > 0:
            forecasts.append(forecasts[-1])
        else:
            forecasts.append(float(train_series.iloc[-1]))

print(forecasts)