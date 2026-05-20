import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
df = df.resample('W-SUN').asfreq()
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].replace(0.0, np.nan)
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# Train/test split
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Feature engineering for training
lags = 52
train_values = train_df['% WEIGHTED ILI'].values

X_train = []
y_train = []
for i in range(lags, len(train_values)):
    X_train.append(train_values[i-lags:i])
    y_train.append(train_values[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Model initialization and training
model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# Recursive forecasting
history = list(train_values[-lags:])
forecasts = []

for _ in range(len(test_df)):
    x_input = np.array(history[-lags:]).reshape(1, -1)
    y_pred = model.predict(x_input)[0]
    forecasts.append(float(y_pred))
    history.append(y_pred)

# Print final forecasts
print(forecasts)