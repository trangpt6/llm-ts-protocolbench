import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Reproducibility not strict for ExponentialSmoothing, but set seed for any random initialization if present
np.random.seed(42)

# Read and prepare data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.sort_values('Month', inplace=True)
df.set_index('Month', inplace=True)
series = df['Passengers'].astype(float)

# Split sizes from Turn 0
train_size = 115
test_size = 29
train = series.iloc[:train_size]
test = series.iloc[train_size:]

# Hyperparameters from Turn 2
model_config = {
    'trend': 'add',
    'seasonal': 'mul',
    'seasonal_periods': 12,
    'damped_trend': False
}

# Rolling one-step ahead forecasting with retraining at each step
history = train.copy()
forecasts = []

for i in range(len(test)):
    # Fit model on current history (all available past data)
    model = ExponentialSmoothing(history,
                                 trend=model_config['trend'],
                                 seasonal=model_config['seasonal'],
                                 seasonal_periods=model_config['seasonal_periods'],
                                 damped_trend=model_config['damped_trend'])
    model_fit = model.fit(disp=False)
    
    # Forecast the next step (horizon=1)
    yhat = model_fit.forecast(steps=1).iloc[0]
    forecasts.append(float(yhat))
    
    # Update history with the true observed value for this test point
    history = pd.concat([history, test.iloc[i:i+1]])

# Output only the final forecast list
print(forecasts)