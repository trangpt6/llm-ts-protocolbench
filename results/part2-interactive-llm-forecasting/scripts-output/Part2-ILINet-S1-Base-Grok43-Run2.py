import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Set random seed for reproducibility
np.random.seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].astype(float)
train = target.iloc[:1016]
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 52), trend='c')
fitted_model = model.fit(disp=False)
forecasts = fitted_model.forecast(steps=254).tolist()
print(forecasts)