import pandas as pd
import numpy as np
import lightgbm as lgb

# Set random seed for reproducibility
np.random.seed(42)

# Read the dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Define train and test sizes
train_size = 158
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Hyperparameters
lags = 12
n_estimators = 300
max_depth = 6
learning_rate = 0.05

# Prepare training data
X_train = []
y_train = []

for i in range(lags, train_size):
    ice_lags = train_df['Ice cream'].iloc[i-lags:i].values
    heater_lags = train_df['Heater'].iloc[i-lags:i].values
    X_train.append(np.concatenate([ice_lags, heater_lags]))
    y_train.append(train_df['Ice cream'].iloc[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train the LightGBM model
model = lgb.LGBMRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=42
)
model.fit(X_train, y_train)

# Recursive forecasting
forecasts = []
current_ice_lags = train_df['Ice cream'].iloc[-lags:].values.tolist()
current_heater_lags = train_df['Heater'].iloc[-lags:].values.tolist()

for i in range(len(test_df)):
    x_input = np.concatenate([current_ice_lags, current_heater_lags]).reshape(1, -1)
    pred = model.predict(x_input)[0]
    forecasts.append(float(pred))
    
    current_ice_lags.append(pred)
    current_ice_lags.pop(0)
    
    current_heater_lags.append(test_df['Heater'].iloc[i])
    current_heater_lags.pop(0)

print(forecasts)