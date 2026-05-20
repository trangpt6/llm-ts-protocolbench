import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df = df.sort_values('DATE').set_index('DATE')

# Create continuous weekly index
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
df = df.reindex(full_idx)

# Replace zero runs in target with NaN (summer non-reporting)
target = '% WEIGHTED ILI'
zero_run_mask = (df[target] == 0).rolling(window=3, min_periods=1).sum() == 3
df.loc[zero_run_mask, target] = np.nan

# 80/20 split
n = len(df)
train_size = int(0.8 * n)
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Rolling forecast with horizon=4, step=1, retrain each step
forecasts = []
history = train[target].copy()

for i in range(len(test)):
    # If history has NaNs, drop them for SARIMA
    hist_clean = history.dropna()
    # Ensure we have enough data
    if len(hist_clean) < 52:
        # fallback: use last observed value
        preds = [hist_clean.iloc[-1]] * 4
    else:
        try:
            model = SARIMAX(hist_clean, order=[1,1,1], seasonal_order=[1,1,1,52], trend='c', enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=200)
            preds = res.forecast(steps=4)
        except:
            preds = [hist_clean.iloc[-1]] * 4
    
    # Ensure we have exactly 4
    if len(preds) < 4:
        preds = list(preds) + [preds[-1]] * (4 - len(preds))
    
    forecasts.extend(preds[:4])
    
    # Update history with the first true value if available
    if i < len(test):
        true_val = test[target].iloc[i]
        # Append true value (may be NaN); update series
        history = pd.concat([history, pd.Series([true_val], index=[test.index[i]])])

# Trim to exact test length (261 steps, but forecasts will be longer because each step gives 4 ahead, overlapping)
# The scenario: rolling window advancing 1 step, forecast horizon 4, so total forecasts = len(test) * 4? Wait.
# Actually: for each step in test set, we forecast 4 steps. But we only need to cover the entire test set? The setup says "forecast horizon: 4" and "window advancement: 1 step per iteration". This means for each time step in the test period, we produce a forecast for the next 4 steps. The total predictions needed to cover the test set is ambiguous. However, typical rolling evaluation: for each step t in test, we forecast t+1..t+4, and we collect only the first forecast (or all if we want multi-step metrics). The instruction: "Generate forecasts that cover the entire test set." Since we have a 261-step test set, if we forecast 4 steps each time, we will have many overlapping forecasts. Usually we take the first step forecast for each time point. Let's re-read: "The total number of predictions must exactly match the length required by the scenario to cover the full test set." The scenario: "Use of ground truth during inference: Enabled (update the model with the true values after each prediction step)". This implies each prediction step produces one forecast (for the next horizon?) No, "Multi-step ahead prediction" with horizon 4 means we produce 4 values per step, but we update with ground truth after each prediction step? The description: "Forecast horizon: 4 (>0) (forecast multiple steps ahead per prediction step)" and "Window advancement: 1 time step per iteration (the window slides by 1, not by H)". So we move one step, forecast 4 ahead, then after obtaining the actual for that step (the first step of the horizon), we use it to update model. Then slide one step. Thus for each time step we produce a 4-step-ahead forecast. The final output list should contain all the forecast values? The instruction "Return the final forecast values directly (as a Python flat list) [y_1, y_2, ..., y_H]" implies H is the length of the test set? Or the number of forecast points? In many benchmarks, they want a list of predictions aligned with the test set. Since horizon is 4, we can produce 261*4 forecasts, but that would be too many. I think the intended interpretation: For each time step in the test set, we forecast the next 4 steps, but we only keep the first prediction (the next step) and then slide. That yields exactly 261 predictions, one per test point. Let's check: "Multi-step ahead prediction" - they want to evaluate multi-step performance? The key: "total number of predictions must exactly match the length required by the scenario to cover the full test set." The test set length is 261. If we use multi-step horizon 4, we need to decide what to output. Many rolling evaluations output only the first step forecast. I'll adopt that: for each step in test, we forecast 4 steps and then update with ground truth, and we collect the first forecast (the one for the next time step). That yields 261 forecasts.

Given the ambiguity, I'll output the first step forecasts only, to match test length.

Thus above code: after computing preds, we take preds[0] as forecast for that step. Then update with true. Let's adjust.

I'll modify to collect only the first forecast per step.

forecasts = []
history = train[target].copy()

for i in range(len(test)):
    hist_clean = history.dropna()
    if len(hist_clean) < 52:
        pred = hist_clean.iloc[-1]
    else:
        try:
            model = SARIMAX(hist_clean, order=[1,1,1], seasonal_order=[1,1,1,52], trend='c', enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False, maxiter=200)
            pred = res.forecast(steps=1).iloc[0]
        except:
            pred = hist_clean.iloc[-1]
    forecasts.append(pred)
    # Update with true value
    if i < len(test):
        true_val = test[target].iloc[i]
        history = pd.concat([history, pd.Series([true_val], index=[test.index[i]])])

print(forecasts)