import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# reproducibility seeds
random.seed(42)
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
df = df[['DATE', '% WEIGHTED ILI']].copy()

train_size = 1040
test_size = 261

train_df = df.iloc[:train_size].copy()

train_df = train_df.set_index('DATE')
full_train_idx = pd.date_range(start=train_df.index.min(), end=train_df.index.max(), freq='7D')
train_df = train_df.reindex(full_train_idx)
train_df['% WEIGHTED ILI'] = train_df['% WEIGHTED ILI'].ffill()

y_train = train_df['% WEIGHTED ILI'].astype(float).values

model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 52),
    trend='c',
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fitted = model.fit(disp=False)

forecasts = np.asarray(fitted.forecast(steps=test_size)).tolist()

print(forecasts)