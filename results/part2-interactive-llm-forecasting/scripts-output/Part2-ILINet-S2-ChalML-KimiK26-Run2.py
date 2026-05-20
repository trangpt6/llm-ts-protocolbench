import random
import numpy as np
import pandas as pd
import lightgbm as lgb

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# parse date column
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

# drop structurally incompatible age columns
df = df.drop(columns=['AGE 25-49', 'AGE 50-64'])

# insert missing weekly date and interpolate target
full_dates = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='7D')
df = pd.DataFrame({'DATE': full_dates}).merge(df, on='DATE', how='left')
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# enforce chronological split from Turn 0
train_df = df[df['DATE'] <= pd.Timestamp('2017-10-01')].copy()
test_df = df[df['DATE'] >= pd.Timestamp('2017-10-08')].copy()

# initialize history and parameters
history = train_df['% WEIGHTED ILI'].values.tolist()
test_values = test_df['% WEIGHTED ILI'].values.tolist()
lags = 13
forecasts = []

# rolling one-step ahead forecast with retraining after each step
for i in range(len(test_values)):
    y_arr = np.array(history)
    n = len(y_arr)
    X_train = np.zeros((n - lags, lags))
    for lag in range(lags):
        X_train[:, lag] = y_arr[lag:n - lags + lag]
    y_train = y_arr[lags:]
    
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=-1
    )
    model.fit(X_train, y_train)
    
    x_pred = y_arr[-lags:].reshape(1, -1)
    pred = float(model.predict(x_pred)[0])
    forecasts.append(pred)
    
    # update with ground truth for next iteration
    history.append(test_values[i])

print(forecasts)