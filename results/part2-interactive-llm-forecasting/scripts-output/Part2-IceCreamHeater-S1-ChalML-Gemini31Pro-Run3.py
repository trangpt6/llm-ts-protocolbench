import pandas as pd
import numpy as np
import lightgbm as lgb

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'
data = df[target].values

# Define split
train_size = 158
test_size = 40
train_data = data[:train_size]

# Feature engineering: Lag features
lags = 12
def create_features(series, n_lags):
    X, y = [], []
    for i in range(n_lags, len(series)):
        X.append(series[i-n_lags:i])
        y.append(series[i])
    return np.array(X), np.array(y)

X_train, y_train = create_features(train_data, lags)

# Initialize and train LightGBM
model = lgb.LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42,
    importance_type='split'
)
model.fit(X_train, y_train)

# Recursive forecasting
forecasts = []
current_batch = list(train_data[-lags:])

for i in range(test_size):
    X_input = np.array(current_batch[-lags:]).reshape(1, -1)
    pred = model.predict(X_input)[0]
    forecasts.append(pred)
    current_batch.append(pred)

print(forecasts)