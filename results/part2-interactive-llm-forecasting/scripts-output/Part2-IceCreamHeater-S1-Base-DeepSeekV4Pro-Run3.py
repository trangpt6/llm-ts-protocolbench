import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv', parse_dates=['Month'], index_col='Month')
df = df.asfreq('MS')
target = df['Ice cream']

# Chronological split: first 158 months train
train = target.iloc[:158]

# Define and fit SARIMA model with fixed hyperparameters
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
fit = model.fit(disp=False)

# Static one-step-ahead recursive forecasts for the entire test set (40 steps)
forecasts = fit.forecast(steps=40).tolist()
print(forecasts)