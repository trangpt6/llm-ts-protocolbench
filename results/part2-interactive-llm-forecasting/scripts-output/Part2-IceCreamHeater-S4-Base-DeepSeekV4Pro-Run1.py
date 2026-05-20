import pandas as pd
import numpy as np
import statsmodels.api as sm

# Reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()
series = df['Ice cream'].astype(np.float64)

# Train/test split: first 158 timesteps train, rest test
train = series.iloc[:158]
test = series.iloc[158:]

# Forecasting setup
block_size = 12
forecast_horizon = 12
forecasts = []
current_train = train.copy()
remaining_test = test.copy()
while len(remaining_test) > 0:
    # Fit SARIMA
    model = sm.tsa.SARIMAX(current_train,
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 12),
                           trend='c')
    fit = model.fit(disp=False)
    # Generate forecasts for horizon=12
    preds = fit.forecast(steps=forecast_horizon)
    needed = min(block_size, len(remaining_test))
    if needed == block_size:
        forecasts.extend(preds.tolist())
        # Update train with true block
        current_train = pd.concat([current_train, remaining_test[:block_size]])
        remaining_test = remaining_test[block_size:]
    else:
        forecasts.extend(preds[:needed].tolist())
        # No further blocks; update not needed but done for completeness
        current_train = pd.concat([current_train, remaining_test])
        remaining_test = remaining_test[needed:]

print(forecasts)