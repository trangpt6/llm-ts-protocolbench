import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seed for reproducibility
np.random.seed(42)

# read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month').sort_index()

# no preprocessing needed per Turn 1

# target column
target_col = 'Passengers'
y = df[target_col]

# train/test split: first 115 observations for training
n_total = len(y)
train_size = 115
y_train_full = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# fixed setup from Turn 2
# rolling update, retrain at every step, forecast horizon 12, window advancement 1
# use ground truth during inference (update model with true values after each prediction step)

# initialize
forecasts = []
history = y_train_full.copy()

# number of prediction steps needed to cover test set with horizon 12
# we need to generate predictions for all 29 test points
# at each step we predict 12 steps ahead, but we only keep predictions that fall within test set
# we advance by 1 each time, retrain, and predict

test_start_idx = train_size  # index in y where test starts
test_end_idx = n_total - 1   # last index in y

# we need to make predictions for all test indices from test_start_idx to test_end_idx
# at step s (0-indexed), we predict indices history_end_idx + 1 to history_end_idx + 12
# we keep predictions for indices in [test_start_idx, test_end_idx]

# determine number of steps: we need to cover up to test_end_idx
# first prediction step: history ends at train_size-1, predict train_size to train_size+11
# last needed prediction step: need to predict test_end_idx, so history_end_idx + h >= test_end_idx
# history_end_idx = train_size - 1 + step
# we need train_size - 1 + step + 1 <= test_end_idx, so step <= test_end_idx - train_size
# but also we need to predict horizon 12 each time, so step goes from 0 to test_end_idx - train_size + some buffer
# actually we need: the last step where we predict must include test_end_idx in its horizon
# history_end_idx = train_size - 1 + step, predict history_end_idx+1 to history_end_idx+12
# need history_end_idx + 1 <= test_end_idx <= history_end_idx + 12
# so step >= test_end_idx - train_size + 1 - 12 = test_end_idx - train_size - 11
# and step <= test_end_idx - train_size

n_test = len(y_test)  # 29
# we need predictions for all 29 test points
# step 0: predict indices 115 to 126 (but 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138 are beyond data)
# actually we only have data up to 143, so step 0 predicts 115-126, all within possible range but 116-126 are test
# we need to run enough steps so that all test indices are covered

# total steps: we need to advance until we've covered all test points
# after step s, history has train_size + s points
# we predict train_size + s to train_size + s + 11
# we need this to cover up to index 143
# train_size + s + 11 >= 143 => s >= 143 - 115 - 11 = 17
# so we need s from 0 to 17, that's 18 steps
# but we also need to ensure we get predictions for all test indices 115-143

# let's track which test indices we've predicted
predicted = {}  # index -> forecast value

max_step = n_test  # 29 steps maximum, but we'll stop when all test indices are covered

for step in range(max_step + 12):  # enough steps to cover all
    current_history_end = train_size + step - 1
    pred_start = current_history_end + 1
    pred_end = current_history_end + 12
    
    # fit SARIMA with fixed hyperparameters
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), trend='n')
    fitted = model.fit(disp=False)
    
    # forecast 12 steps ahead
    pred = fitted.get_forecast(steps=12)
    pred_mean = pred.predicted_mean
    
    # store predictions for test indices
    for h in range(12):
        idx = pred_start + h
        if test_start_idx <= idx <= test_end_idx and idx not in predicted:
            predicted[idx] = pred_mean.iloc[h]
    
    # check if all test indices are covered
    if len(predicted) == n_test:
        break
    
    # rolling update: add true value to history (ground truth enabled)
    if step < n_test:
        next_true = y.iloc[train_size + step]
        history = pd.concat([history, pd.Series([next_true], index=[y.index[train_size + step]])])

# assemble final forecasts in order
forecasts = [predicted[idx] for idx in sorted(predicted.keys())]

print(forecasts)