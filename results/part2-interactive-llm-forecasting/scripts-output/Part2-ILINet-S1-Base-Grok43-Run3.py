import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
non_target_cols = [col for col in df.columns if col != '% WEIGHTED ILI']
df[non_target_cols] = df[non_target_cols].fillna(method='ffill')
target = df['% WEIGHTED ILI']
train = target.iloc[:1020]
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c')
fitted_model = model.fit(disp=False)
forecasts = fitted_model.forecast(steps=256).tolist()
print(forecasts)