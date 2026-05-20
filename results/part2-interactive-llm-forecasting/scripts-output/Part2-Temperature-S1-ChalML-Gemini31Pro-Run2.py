import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')

# Apply preprocessing steps
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index.freq = 'D'

# Define target and split sizes
target_col = 'Daily minimum temperatures'
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

# Extract training series
train_series = df[target_col].iloc[:train_size].values

# Prepare training data with 14 lags
lags = 14
X_train = []
y_train = []
for i in range(lags, len(train_series)):
    X_train.append(train_series[i-lags:i])
    y_train.append(train_series[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Initialize and train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# Perform recursive one-step ahead forecasting
forecasts = []
current_history = list(train_series[-lags:])

for _ in range(test_size):
    x_input = np.array(current_history[-lags:]).reshape(1, -1)
    pred = model.predict(x_input)[0]
    forecasts.append(float(pred))
    current_history.append(float(pred))

# Print final forecasts
print(forecasts)