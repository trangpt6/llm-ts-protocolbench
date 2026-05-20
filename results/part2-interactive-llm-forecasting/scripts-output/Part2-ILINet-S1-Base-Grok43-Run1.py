import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target_col = '% WEIGHTED ILI'
series = df[target_col].copy().astype(float)
for i in range(len(series)):
    if series.iloc[i] == 0:
        non_zeros = []
        j = i - 1
        while len(non_zeros) < 4 and j >= 0:
            if series.iloc[j] != 0:
                non_zeros.append(series.iloc[j])
            j -= 1
        if len(non_zeros) > 0:
            series.iloc[i] = np.median(non_zeros)
        else:
            series.iloc[i] = 0.0
df[target_col] = series
train_size = 1016
train_data = df[target_col].iloc[:train_size]
model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c')
fitted_model = model.fit(disp=False)
forecasts = fitted_model.forecast(steps=254).tolist()
print(forecasts)