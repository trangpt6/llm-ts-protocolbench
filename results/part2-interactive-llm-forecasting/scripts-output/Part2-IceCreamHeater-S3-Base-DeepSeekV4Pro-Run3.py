import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility
np.random.seed(42)

# read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv', parse_dates=['Month'], index_col='Month')
target = df['Ice cream']

# train/test split
train = target.iloc[:158]
test = target.iloc[158:]

# fixed hyperparameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
trend = 'c'

# rolling origin forecast with retraining
history = train.values.tolist()
forecasts = []

for i in range(len(test)):
    # fit SARIMAX on current history
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend)
    fitted = model.fit(disp=False)
    # forecast 1 step ahead (horizon=1, but we only need the next step)
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(pred)
    # update history with true value
    true_val = test.iloc[i]
    history.append(true_val)

print(forecasts)