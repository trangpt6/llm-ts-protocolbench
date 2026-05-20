import pandas as pd
import numpy as np
import statsmodels.api as sm

# Set random seed for reproducibility
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], index_col='Date')
series = df['Daily minimum temperatures'].astype(float)

# Chronological split: 80% training, 20% test
split_point = int(0.8 * len(series))
# Actual split point matches Turn 0: 2921

# Fixed model hyperparameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 365)
trend = 'n'

forecasts = []

# Rolling forecast with horizon 7, sliding by 1, retrain after each step using ground truth
for i in range(len(series) - split_point):
    end_train = split_point + i  # index of last training point (exclusive upper bound)
    train_data = series.iloc[:end_train]
    # Fit SARIMA model
    model = sm.tsa.SARIMAX(train_data, order=order, seasonal_order=seasonal_order, trend=trend)
    fit = model.fit(disp=False)
    # Forecast next 7 steps, we only need the first step for the final test set prediction
    pred = fit.forecast(steps=7)
    forecasts.append(pred.iloc[0])

# Output the flat list of forecasts covering the entire test set
print(forecasts)