import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
train_size = 2920
train = df.iloc[:train_size].copy()
non_zero_median = train[train['Daily minimum temperatures'] != 0]['Daily minimum temperatures'].median()
train.loc[train['Daily minimum temperatures'] == 0, 'Daily minimum temperatures'] = non_zero_median
y_train = train['Daily minimum temperatures'].values
model = SARIMAX(y_train, order=(1, 0, 1), seasonal_order=(1, 1, 1, 365), trend='n')
fitted_model = model.fit(disp=False)
forecasts = fitted_model.forecast(steps=730).tolist()
print(forecasts)