import pandas as pd
import numpy as np
import statsmodels.api as sm

# Read data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'], index_col='DATE')

# Preprocessing: reindex to complete weekly range, drop rows where target is NaN
full_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_dates)
df = df.dropna(subset=['% WEIGHTED ILI'])
y = df['% WEIGHTED ILI'].astype(float)

# Chronological split
train = y.iloc[:1050]
test = y.iloc[1050:]

# Fixed SARIMA model
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)
model = sm.tsa.SARIMAX(train, order=order, seasonal_order=seasonal_order, trend='c')

# Fit model
np.random.seed(42)
res = model.fit(disp=False)

# Recursive one-step ahead forecast for entire test period (no true values used)
forecast_result = res.get_forecast(steps=len(test), dynamic=True)
forecasts = forecast_result.predicted_mean.tolist()

# Output final forecast list
print(forecasts)