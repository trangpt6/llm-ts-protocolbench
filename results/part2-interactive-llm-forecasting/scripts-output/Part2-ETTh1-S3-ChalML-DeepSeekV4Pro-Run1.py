import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Identify constant-value blocks at end of months (all 24 hours identical)
df['hour'] = df.index.hour
month_end_const = df.groupby([df.index.year, df.index.month, df.index.day])['OT'].transform('nunique') == 1
df.loc[month_end_const & (df['hour'] == 23), 'OT'] = np.nan  # replace the whole day with NaN by flagging last hour; simpler: mark all hours on that day as NaN if that day is entirely constant. We'll do: for each day, if all values equal and it's last day of month, set all to NaN.
# Better: create a mask for days where all OT values are identical and the next day is 1st of next month.
df['date_day'] = df.index.date
daily_constant = df.groupby('date_day')['OT'].transform('nunique') == 1
next_day_is_1st = df.index.shift(-1).floor('D') != df.index.floor('D')
# Actually, to detect the last day of the month: if next day's day is 1. But careful: the last day may be constant and not just at end-of-month? We'll mark only days where all 24 values are identical AND the date is the last day of the month.
is_month_end = df.index.day == (df.index + pd.offsets.MonthEnd(0)).day
df.loc[daily_constant & is_month_end, 'OT'] = np.nan
df.drop(['hour','date_day'], axis=1, inplace=True)

# Split index at 14016 (80%)
train_idx = df.iloc[:14016].index
test_idx = df.iloc[14016:].index

target = 'OT'
lags = 12
horizon = 24

def create_lag_features(series, lags):
    # From 1D numpy array of target series, create matrix X of lag features for direct multi-step: use the last lags values as features, and predict horizon steps ahead. For each sample, input = last lags values, output = future horizon values (vector). We'll use multi-output regression. To create samples, for each time t where we have lags and future horizon known, we construct a sample.
    n = len(series)
    X = []
    y = []
    for i in range(lags, n - horizon + 1):
        X.append(series[i - lags : i])
        y.append(series[i : i + horizon])
    return np.array(X), np.array(y)

y_full = df[target].values
# Prepare initial training set from train period (including NaNs)
train_y = y_full[:14016]

# Rolling forecasting with retraining at each step
predictions = []
current_train_end = len(train_y) - 1  # index of last training observation
for step in range(len(test_idx)):
    # Retrain model on all available true data up to current_train_end (which includes any true values from test set already revealed)
    # The true data includes all timestamps up to current_train_end (which may be past test points that have been observed as ground truth)
    y_avail = y_full[:current_train_end + 1].copy()
    # Create lag features for training from this series
    X_train, y_train_multi = create_lag_features(y_avail, lags)
    # Remove any NaN rows caused by missing values
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train_multi).any(axis=1)
    X_train = X_train[valid]
    y_train_multi = y_train_multi[valid]
    
    # Model for direct multi-step
    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, objective='reg:squarederror', verbosity=0)
    model.fit(X_train, y_train_multi)
    
    # Get the last lags from current available data (current_train_end) to forecast next horizon
    last_lags = y_avail[-lags:]
    if np.isnan(last_lags).any():
        # If any NaN in last_lags, cannot forecast; fill with last valid value (but shouldn't happen often)
        last_valid = y_avail[~np.isnan(y_avail)][-lags:]
        if len(last_valid) < lags:
            last_valid = np.repeat(last_valid[-1], lags)
        last_lags = last_valid
    X_pred = last_lags.reshape(1, -1)
    y_pred = model.predict(X_pred).flatten()  # shape (horizon,)
    
    # Append the first predicted value (the immediate next step) as the forecast for this step; we will slide by 1.
    # But the scenario says: forecast horizon 24, then advance window by 1 time step, and use the true value of that step to update model. So at each step, we predict 24 steps ahead, but we only need to collect the 1-step-ahead forecast? Wait: The output should be forecasts covering the entire test set. The test set has length 3504. At each time step, we predict 24 steps ahead, but we are sliding by 1. Which forecasts do we keep? Typically in rolling forecasting with horizon=24 and step=1, you would keep all 24 forecasts and they overlap? But the final required forecast list should have length test_size. Since we advance by 1, we can either accumulate the 1-step ahead forecasts (the first element of each 24-step prediction) to get a sequence of length test_size, which is the usual approach for multi-step rolling evaluation with direct strategy. But the instruction: "The model predicts the next 24 hours, then the window advances by 1 hour using the true observed value to update the model before making the next 24-hour forecast." So after each forecast, we get the true value for the next hour and add it to the training data, then retrain. The forecasts we output could be the concatenation of all predicted values? That would exceed test length. Typically for a fixed test set of length T, we produce a forecast for each test point by using the model trained up to the previous point, and the "multi-step" refers to the horizon of prediction in each iteration, but the evaluation is done on the 1-step-ahead predictions (or the full horizon)? The prompt: "Generates a single flat list of forecasts covering the entire test set." So we need exactly one forecast per test time step. The logical approach: For each test time t (starting from the first test time), we use data up to t-1 to train, predict the value at t (which is the 1-step-ahead). Then we incorporate the true value at t, retrain, and predict t+1. That is a 1-step-ahead rolling forecast, not using the horizon 24? But the horizon is fixed at 24. However, the setup says "Forecast horizon: 24 (>0) (forecast multiple steps ahead per prediction step)". That means at each iteration we produce a 24-step-ahead forecast. The rolling window advances by 1 step after receiving the true value. So we could output all 24-step-ahead predictions and they would overlap, but we need a single vector of length test_size. It might be that we are to output the 1-step-ahead prediction from each 24-step forecast, effectively using the model as a step-ahead predictor. This is a common scenario in direct multi-step forecasting evaluation: you train a model to directly predict h steps ahead, but then to generate forecasts for the test set, you iterate, using the 1-step prediction at each step. Alternatively, we could treat the entire 24-step ahead as the forecast for the next 24 hours, but then sliding by 1 would cause overlapping predictions. However, the typical benchmark expects a single forecast per test point. I'll assume we need to produce a forecast for the first test time step using the model trained up to the last training point, then after observing that true value, we update and forecast the next test step, and so on. So we only keep the first element of each 24-step prediction. That effectively makes horizon irrelevant for output but still used in training. That makes sense.

Thus, at each step, after fitting, we predict the next 24 steps, but we only record the prediction for the immediate next step (the first element of y_pred). Then we advance current_train_end by 1, incorporate the true value of that step (from test set), retrain, and repeat. The collected predictions will have length len(test_idx). So that matches.

I'll implement accordingly.

Then at the end, print the list of predictions. I'll use `print(predictions)`.

Make sure script runs without errors, imports only needed libraries.

Now I'll output the script without any markdown or extra text.import pandas as pd
import numpy as np
from xgboost import XGBRegressor

np.random.seed(42)

df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

# Preprocessing: mark constant-value blocks at month-end as NaN
df['day'] = df.index.date
daily_constant = df.groupby('day')['OT'].transform('nunique') == 1
is_month_end = df.index.day == (df.index + pd.offsets.MonthEnd(0)).day
df.loc[daily_constant & is_month_end, 'OT'] = np.nan
df.drop('day', axis=1, inplace=True)

train_size = 14016
y_full = df['OT'].values

def create_lag_features(series, lags, horizon):
    X, y = [], []
    for i in range(lags, len(series) - horizon + 1):
        X.append(series[i - lags : i])
        y.append(series[i : i + horizon])
    return np.array(X), np.array(y)

lags = 12
horizon = 24
predictions = []
current_end = train_size - 1  # last training index

for step in range(len(df) - train_size):
    # Use all available data up to current_end
    y_avail = y_full[:current_end + 1].copy()
    X_train, y_train = create_lag_features(y_avail, lags, horizon)
    valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train).any(axis=1)
    X_train = X_train[valid]
    y_train = y_train[valid]

    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1,
                         random_state=42, objective='reg:squarederror', verbosity=0)
    model.fit(X_train, y_train)

    last_lags = y_avail[-lags:]
    if np.isnan(last_lags).any():
        last_valid = y_avail[~np.isnan(y_avail)]
        last_valid = last_valid[-lags:]
        if len(last_valid) < lags:
            last_valid = np.pad(last_valid, (lags - len(last_valid), 0), 'edge')
        last_lags = last_valid
    X_pred = last_lags.reshape(1, -1)
    y_pred = model.predict(X_pred).flatten()
    # Record the one-step-ahead forecast (first element)
    predictions.append(y_pred[0])
    # Advance and incorporate ground truth
    current_end += 1

print(predictions)