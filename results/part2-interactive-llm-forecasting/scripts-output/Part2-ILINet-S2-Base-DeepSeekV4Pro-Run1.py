import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility in cases where solver uses random init
np.random.seed(42)

# Load and preprocess data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.drop(columns=['AGE 25-49'], inplace=True)
df.set_index('DATE', inplace=True)
df.sort_index(inplace=True)
target = df['% WEIGHTED ILI'].astype(float)

# Chronological split: first 80% train, remaining 20% test
n = len(target)
train_size = int(0.8 * n)
y_train = target.iloc[:train_size]
y_test = target.iloc[train_size:]

# Fixed SARIMA hyperparameters
order = (2, 0, 2)
seasonal_order = (1, 1, 1, 52)
trend = 'n'

# Rolling one-step ahead forecasting with retraining after each step
history = y_train.copy()
forecasts = []

for i in range(len(y_test)):
    # Fit SARIMA on current history
    model = SARIMAX(history, order=order, seasonal_order=seasonal_order, trend=trend)
    try:
        fit = model.fit(disp=False)
    except Exception:
        # If convergence fails, fallback to simple naive forecast as last resort
        forecasts.append(history.iloc[-1])
        true_val = y_test.iloc[i]
        history = pd.concat([history, pd.Series([true_val], index=[y_test.index[i]])])
        continue
    # One-step forecast
    pred = fit.forecast(steps=1)
    forecasts.append(pred.iloc[0])
    # Update history with true value
    true_val = y_test.iloc[i]
    history = pd.concat([history, pd.Series([true_val], index=[y_test.index[i]])])

print(forecasts)