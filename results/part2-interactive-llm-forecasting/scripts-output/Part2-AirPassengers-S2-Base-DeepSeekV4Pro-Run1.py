import pandas as pd
import numpy as np
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
df = df.asfreq('MS')  # ensure monthly start frequency
series = df['Passengers']

# fixed chronological split
train_size = 115
train = series.iloc[:train_size].copy()
test = series.iloc[train_size:].copy()

# fixed hyperparameters
hyperparams = {
    'trend': 'add',
    'seasonal': 'mul',
    'seasonal_periods': 12,
    'damped_trend': False,
}

# rolling one-step ahead forecast with retraining after each step
forecasts = []
current_series = train.copy()

for i in range(len(test)):
    # fit model on current series
    model = ExponentialSmoothing(
        current_series.astype(float).values,
        trend=hyperparams['trend'],
        seasonal=hyperparams['seasonal'],
        seasonal_periods=hyperparams['seasonal_periods'],
        damped_trend=hyperparams['damped_trend'],
        initialization_method='estimated',
    )
    fitted = model.fit(optimized=True, use_boxcox=False, remove_bias=False)
    # forecast next step
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(pred)
    # update current series with the true test value (ground truth available)
    true_val = test.iloc[i]
    current_series = pd.concat([current_series, pd.Series([true_val], index=[test.index[i]])])

print(forecasts)