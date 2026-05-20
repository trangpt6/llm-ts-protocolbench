import random
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# read the raw csv file
df = pd.read_csv(r'../../../data/ILINet.csv')

# parse dates and establish weekly sunday frequency
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
df = df.set_index('DATE')
df = df.asfreq('W-SUN')

# apply linear interpolation to fill the single missing timestamp
df = df.interpolate(method='linear')

# define target and fixed chronological split boundaries from turn 0
target_col = '% WEIGHTED ILI'
train_df = df.loc[:'2017-10-01']
test_df = df.loc['2017-10-08':]

train_series = train_df[target_col].to_numpy()
test_series = test_df[target_col].to_numpy()

# construct lag features from training target only
lags = 52
X_train = []
y_train = []
for i in range(lags, len(train_series)):
    X_train.append(train_series[i - lags:i])
    y_train.append(train_series[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# initialize xgboost with fixed hyperparameters
model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# recursive one-step ahead forecasting without ground truth updates
history = list(train_series)
forecasts = []
for _ in range(len(test_series)):
    x_input = np.array(history[-lags:]).reshape(1, -1)
    y_pred = model.predict(x_input)[0]
    forecasts.append(float(y_pred))
    history.append(float(y_pred))

print(forecasts)