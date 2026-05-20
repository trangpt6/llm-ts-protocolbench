import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# Set random seeds for reproducibility
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# Convert DATE to datetime and set as index
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# Resample to weekly frequency to align missing weeks
df = df.resample('W-SUN').asfreq()

# Impute missing values in the primary target using linear interpolation
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# Keep only the primary target
df = df[['% WEIGHTED ILI']]

# Define train/test split
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train_data = df.iloc[:train_size]['% WEIGHTED ILI'].values
test_data = df.iloc[train_size:]['% WEIGHTED ILI'].values

# Hyperparameters
lags = 52
n_estimators = 500
max_depth = 6
learning_rate = 0.05

# Prepare training data
X_train = []
y_train = []
for i in range(lags, len(train_data)):
    X_train.append(train_data[i-lags:i])
    y_train.append(train_data[i])
X_train = np.array(X_train)
y_train = np.array(y_train)

# Initialize and train the model
model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
model.fit(X_train, y_train)

# Recursive forecasting
forecasts = []
history = list(train_data[-lags:])

for _ in range(len(test_data)):
    X_test = np.array(history[-lags:]).reshape(1, -1)
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    history.append(pred)

# Print the final forecast list
print(forecasts)