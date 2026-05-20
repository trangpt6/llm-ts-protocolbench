import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE').sort_index()

# Preprocessing: Insert missing timestamp 2002-01-06
full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_range)
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# Preprocessing: Handle exogenous missing values
# AGE 25-49 and AGE 50-64 are missing in early years. Fill with 0 as per Turn 1.
df['AGE 25-49'] = df['AGE 25-49'].fillna(0)
df['AGE 50-64'] = df['AGE 50-64'].fillna(0)

# Define split based on Turn 0
total_timesteps = 1305
train_size = 1044
test_size = 261

target_col = '% WEIGHTED ILI'
series = df[target_col].values

# Fixed Setup: SARIMA with rolling update, retraining every step, horizon 1
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
forecasts = []

# Rolling forecast loop
for i in range(test_size):
    # Current training set includes all data up to the current test point
    current_train = series[:train_size + i]
    
    # Retrain model at every step
    model = SARIMAX(current_train, 
                    order=order, 
                    seasonal_order=seasonal_order, 
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    
    # One-step ahead forecast
    pred = model_fit.forecast(steps=1)[0]
    forecasts.append(float(pred))

# Print final forecast list
print(forecasts)