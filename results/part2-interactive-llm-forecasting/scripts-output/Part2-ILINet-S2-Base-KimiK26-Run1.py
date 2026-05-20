import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# set random seed for reproducibility
np.random.seed(42)

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])

for col in ['AGE 25-49', 'AGE 25-64']:
    if col in df.columns:
        df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

y = df['% WEIGHTED ILI'].astype(float)
train = y.iloc[:1045]
test = y.iloc[1045:]

history = train.copy()
forecasts = []

for i in range(len(test)):
    model = SARIMAX(history, order=(2, 0, 2), seasonal_order=(1, 1, 1, 52), trend='n')
    result = model.fit(disp=False)
    pred = float(result.forecast(steps=1).iloc[0])
    forecasts.append(pred)
    history = pd.concat([history, pd.Series([test.iloc[i]])], ignore_index=True)

print(forecasts)