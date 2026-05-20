import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# Set random seeds
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Target
y = df['Ice cream'].values

# Split
train_size = 158
y_train = y[:train_size]
y_test = y[train_size:]

# Initialize history and forecasts
history = list(y_train)
forecasts = []
block_size = 12

# Rolling block forecast
for i in range(0, len(y_test), block_size):
    steps = min(block_size, len(y_test) - i)
    model = sm.tsa.SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='c')
    fitted_model = model.fit(disp=False)
    fc = fitted_model.forecast(steps=steps)
    forecasts.extend(fc.tolist())
    history.extend(y_test[i:i+steps])

print(forecasts)