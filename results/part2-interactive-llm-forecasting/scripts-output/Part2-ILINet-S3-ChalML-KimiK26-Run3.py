import random
import numpy as np
import pandas as pd
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)

target_col = '% WEIGHTED ILI'
data = df[target_col].values

total = len(data)
train_size = int(0.8 * total)
train_data = data[:train_size]
test_data = data[train_size:]

lags = 13
H = 4

def create_lagged_features(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

X_train, y_train = create_lagged_features(train_data, lags)

model = lgb.LGBMRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

forecasts = []
current_window = list(train_data)

for i in range(len(test_data)):
    X_pred = np.array([current_window[-lags:]])
    preds = []
    rec_input = X_pred.flatten().tolist()
    for step in range(H):
        pred = model.predict(np.array([rec_input[-lags:]]))[0]
        preds.append(pred)
        rec_input.append(pred)
    forecasts.extend(preds)

    # Use ground truth for retraining
    true_val = test_data[i]
    current_window.append(true_val)
    if len(current_window) > lags + 50:
        X_retrain, y_retrain = create_lagged_features(current_window, lags)
        model.fit(X_retrain, y_retrain)

# Trim to exact test set length if needed (should be exactly test_size * H? No, we produce H per step, total test_size steps)
# Actually we generated H per step, but we only need total test_size forecasts? Wait: With rolling update and horizon 4, each step we forecast 4 ahead. The test set has 261 steps. We need to cover entire test set, meaning we produce 261 forecasts? Or produce 4*261? The setup says "Forecast horizon: 4" and "window advancement: 1". At each iteration we forecast next 4 steps. But to cover the entire test set, we need to output the first forecast of each horizon? The instruction ambiguous. Usually rolling multi-step with horizon H and advancement 1, we produce one forecast per step (the first step of the horizon) and then slide. But the definition says "multi-step ahead prediction" and "forecast horizon: 4". To cover the full test set, we need to generate forecasts for each time point in the test set. Common approach: for each step, forecast the next H values, but only the first value is used for comparison, then slide. But the instruction says "generate forecasts that cover the entire test set" and "total number of predictions must exactly match the length required by the scenario". The scenario: "Forecast horizon: 4 (>0) (forecast multiple steps ahead per prediction step)" and "Window advancement: 1". That suggests we produce one forecast per step (the first step of the horizon). Actually typical rolling origin: at each step, forecast next H steps, but only the first is evaluated, then we move by 1. So total forecasts = length of test set. But then why horizon 4? Possibly evaluate multiple horizons. The instruction says "multi-step ahead prediction", so likely we need to output all H forecasts for each step? But then total predictions would be test_size * H. Let's re-read: "Forecasting type: Multi-step ahead prediction" and "Forecast horizon: 4". Then "Window advancement: 1 time step per iteration". It doesn't specify whether we output all H or just the first. However, the typical benchmark expects one forecast per time step (the one-step-ahead forecast) with a model that internally uses multi-step strategy (recursive). But the "use of ground truth during inference: Enabled" suggests we update after each step. So I think we need to output one forecast per test point. But the horizon 4 is used to generate predictions recursively, but only the first forecast is retained. However, the instruction says "the total number of predictions must exactly match the length required by the scenario". If we output 4 per step, that's too many. I will assume we need to output one forecast per test point (the first step of the horizon). But to be safe, I'll output a list of length equal to test set (261). I'll implement: at each iteration, forecast 4 steps, take the first as the forecast for that step. Then use true value to retrain.

I'll adjust code accordingly: after generating preds list of 4, append preds[0] to forecasts.

Then trim to test_size.

Let me produce final script.
```

```python
import random
import numpy as np
import pandas as pd
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)

target_col = '% WEIGHTED ILI'
data = df[target_col].values

total = len(data)
train_size = int(0.8 * total)
train_data = data[:train_size]
test_data = data[train_size:]

lags = 13
H = 4

def create_lagged_features(series, lags):
    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

X_train, y_train = create_lagged_features(train_data, lags)

model = lgb.LGBMRegressor(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

forecasts = []
current_window = list(train_data)

for i in range(len(test_data)):
    X_pred = np.array([current_window[-lags:]])
    rec_input = X_pred.flatten().tolist()
    for step in range(H):
        pred = model.predict(np.array([rec_input[-lags:]]))[0]
        rec_input.append(pred)
    # Only use first forecast for this step
    forecasts.append(rec_input[lags])  # first predicted value
    # Update with true value
    true_val = test_data[i]
    current_window.append(true_val)
    if len(current_window) >= lags + 2:
        X_retrain, y_retrain = create_lagged_features(current_window, lags)
        model.fit(X_retrain, y_retrain)

print(forecasts)