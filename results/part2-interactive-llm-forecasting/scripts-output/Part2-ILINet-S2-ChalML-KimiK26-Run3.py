import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

# set random seeds for reproducibility
np.random.seed(42)

# read dataset from local file
df = pd.read_csv(r'../../../data/ILINet.csv')

# parse date column
df['DATE'] = pd.to_datetime(df['DATE'])

# create complete weekly date range and reindex to regular frequency
full_range = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='7D')
df = df.set_index('DATE').reindex(full_range).reset_index()
df = df.rename(columns={'index': 'DATE'})

# interpolate target for the inserted missing timestamp only
target_col = '% WEIGHTED ILI'
df[target_col] = df[target_col].astype(float).interpolate(method='linear')

# enforce exact chronological train/test split from Turn 0
train_df = df[df['DATE'] <= '2017-10-01'].copy()
test_df = df[df['DATE'] >= '2017-10-08'].copy()

# extract target values
hist = train_df[target_col].tolist()
test_vals = test_df[target_col].values
forecasts = []

# rolling one-step ahead forecast: retrain after each ground-truth update
for step in range(len(test_vals)):
    y_hist = np.array(hist)
    X = []
    y = []
    for i in range(13, len(y_hist)):
        X.append(y_hist[i-13:i])
        y.append(y_hist[i])
    X = np.array(X)
    y = np.array(y)
    model = LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    x_pred = y_hist[-13:].reshape(1, -1)
    pred = float(model.predict(x_pred)[0])
    forecasts.append(pred)
    hist.append(float(test_vals[step]))

print(forecasts)