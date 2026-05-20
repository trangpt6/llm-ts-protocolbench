import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set random seed for reproducibility
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# Parse dates and set index
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.set_index('DATE').sort_index()

# Drop entirely empty columns (AGE 25-49, AGE 25-64)
df = df.drop(columns=['AGE 25-49', 'AGE 25-64'], errors='ignore')

# Target column
target_col = '% WEIGHTED ILI'
y = df[target_col].astype(float)

# Train/test split: 80/20 chronological
train_size = int(0.8 * len(y))  # 1050
test_size = len(y) - train_size  # 263

# Fixed SARIMA parameters
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 52)
trend = 'c'

# Rolling forecast with horizon 4, sliding 1, retrain at every step
forecasts = []
current_train = y.iloc[:train_size].copy()

for i in range(test_size):
    # Fit SARIMA on current training data
    model = SARIMAX(current_train, order=order, seasonal_order=seasonal_order, trend=trend)
    fitted = model.fit(disp=False)
    
    # Forecast 4 steps ahead
    pred = fitted.forecast(steps=4)
    
    # Store only the first forecast (1-step ahead) for the current test point
    forecasts.append(pred.iloc[0])
    
    # Append the true test value to training data for next iteration
    true_next = y.iloc[train_size + i]
    current_train = pd.concat([current_train, pd.Series([true_next], index=[y.index[train_size + i]])])
    
    # Optional: avoid memory blow up by keeping only last part, but SARIMA fits may require full history,
    # so we keep all. This is manageable for 1313 points.

# Print the final forecast list
print(forecasts)