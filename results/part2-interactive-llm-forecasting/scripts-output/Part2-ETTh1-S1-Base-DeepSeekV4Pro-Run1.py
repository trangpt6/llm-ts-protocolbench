import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
df = df.asfreq('H')

# Preprocessing: Replace end-of-month constant-value blocks with NaN
end_of_month_dates = [
    '2016-07-31', '2016-08-31', '2016-10-31', '2016-12-31',
    '2017-01-31', '2017-03-31', '2017-05-31', '2017-07-31',
    '2017-08-31', '2017-10-31', '2017-12-31', '2018-01-31',
    '2018-03-31', '2018-05-31'
]
for d in end_of_month_dates:
    mask = df.index.normalize() == pd.Timestamp(d)
    df.loc[mask, 'OT'] = np.nan

# Split into train and test (80/20 chronological)
train_size = int(0.8 * len(df))
train = df.iloc[:train_size]
test = df.iloc[train_size:]

# Fit SARIMA model
model = SARIMAX(
    train['OT'],
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 24),
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted = model.fit(disp=False, maxiter=500)

# Forecast recursively one-step ahead for entire test set
forecast_result = fitted.get_forecast(steps=len(test))
forecasts = forecast_result.predicted_mean.tolist()

print(forecasts)