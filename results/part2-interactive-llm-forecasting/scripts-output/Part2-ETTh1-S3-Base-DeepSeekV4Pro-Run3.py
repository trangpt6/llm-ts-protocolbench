import pandas as pd
import numpy as np
import random
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'], index_col='date')

# Preprocessing: detect constant-value days and replace with NaN
df['day'] = df.index.date
constant_mask = df.groupby('day')['OT'].transform('nunique') == 1
df.loc[constant_mask, 'OT'] = np.nan

# Interpolate linearly to fill missing values for regular time series
df['OT'] = df['OT'].interpolate(method='linear')

# Drop the temporary day column
df = df.drop(columns='day')

# Split into train and test
total_rows = len(df)
train_size = int(0.8 * total_rows)
train = df.iloc[:train_size]['OT']
test = df.iloc[train_size:]['OT']

# Define SARIMA hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)

# Initial model fitting on training data
model = SARIMAX(train, order=order, seasonal_order=seasonal_order, trend='c')
result = model.fit(disp=False)

# Rolling forecast
forecasts = []
current_train = train.copy()

for i in range(len(test)):
    # Forecast 24 steps ahead
    forecast_obj = result.get_forecast(steps=24)
    pred_mean = forecast_obj.predicted_mean
    # Keep only the 1-step ahead forecast (first value)
    forecasts.append(pred_mean.iloc[0])

    # Update training data with true test value for this step
    current_train = pd.concat([current_train, pd.Series(test.iloc[i], index=[test.index[i]])])

    # Retrain the model on the updated training data
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend='c')
    result = model.fit(disp=False)

# Output forecasts as a flat list
print(forecasts)