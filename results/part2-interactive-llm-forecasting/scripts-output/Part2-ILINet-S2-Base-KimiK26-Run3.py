import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

if 'AGE 25-49' in df.columns:
    df['AGE 25-49'] = df['AGE 25-49'].ffill()
if 'AGE 50-64' in df.columns:
    df['AGE 50-64'] = df['AGE 50-64'].ffill()

target_col = '% WEIGHTED ILI'
train_size = int(0.8 * len(df))
train_series = df[target_col].iloc[:train_size].values
test_series = df[target_col].iloc[train_size:].values

history = train_series.tolist()
forecasts = []

for t in range(len(test_series)):
    model = SARIMAX(
        history,
        order=(2, 0, 2),
        seasonal_order=(1, 1, 1, 52),
        trend='n',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    result = model.fit(disp=False, cov_type='none')
    pred = result.forecast(steps=1)[0]
    forecasts.append(float(pred))
    history.append(float(test_series[t]))

print(forecasts)