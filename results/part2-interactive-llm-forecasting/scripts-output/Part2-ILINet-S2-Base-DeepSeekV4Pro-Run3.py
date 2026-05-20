import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# set seed for reproducibility
np.random.seed(42)

# read data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'], dayfirst=False)

# store original length for split
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)

# get split date from original data (chronological split)
split_date = df.iloc[train_size - 1]['DATE']

# preprocessing: remove rows where all numeric columns are zero (summer shutdowns)
numeric_cols = [col for col in df.columns if col != 'DATE']
mask_all_zero = (df[numeric_cols] == 0).all(axis=1)
df = df[~mask_all_zero].copy()

# split using the original split date
train = df[df['DATE'] <= split_date].copy()
test = df[df['DATE'] > split_date].copy()

# extract target series
y_train = train['% WEIGHTED ILI'].values
y_test = test['% WEIGHTED ILI'].values

# rolling one-step ahead forecasting
forecasts = []
history = y_train.tolist()
for i in range(len(y_test)):
    model = SARIMAX(history, order=(2,0,2), seasonal_order=(1,1,1,52), trend='n')
    fitted = model.fit(disp=False, maxiter=200, method='lbfgs')
    pred = fitted.forecast(steps=1)[0]
    forecasts.append(pred)
    # update history with true observation
    history.append(y_test[i])

print(forecasts)