import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
series = df['Ice cream'].values

# Chronological split
train_size = 158
train = series[:train_size]
test = series[train_size:]

# Fixed SARIMA parameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)

# Rolling forecast with horizon 12, step 1, retrain every step, use ground truth updates
history = list(train)
forecasts = []

for t in range(len(test)):
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend='c')
    fitted = model.fit(disp=False)
    # Multi-step ahead forecast: horizon 12
    pred = fitted.forecast(steps=12)
    # We only need the next step (t+1 from current origin) to cover the test set step by step
    next_forecast = pred[0]
    forecasts.append(next_forecast)
    # Update history with actual test value for next retraining
    history.append(test[t])

print(forecasts)