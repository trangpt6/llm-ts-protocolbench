import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target = 'Ice cream'
data = df[target].values

# Define split point based on Turn 0
train_size = 158
test_size = 40
lags = 6

# Feature engineering function
def create_features(series, n_lags):
    X, y = [], []
    for i in range(len(series) - n_lags):
        X.append(series[i:i + n_lags])
        y.append(series[i + n_lags])
    return np.array(X), np.array(y)

# Forecasting loop with rolling update and retraining
forecasts = []
current_data = list(data[:train_size])

for i in range(test_size):
    # Prepare training data from current available history
    X_train, y_train = create_features(np.array(current_data), lags)
    
    # Initialize and train XGBoost model with fixed hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    
    # Prepare input for one-step ahead prediction (last 'lags' observations)
    X_test = np.array(current_data[-lags:]).reshape(1, -1)
    
    # Predict next step
    pred = model.predict(X_test)[0]
    forecasts.append(float(pred))
    
    # Update history with ground truth for next iteration (rolling update)
    current_data.append(data[train_size + i])

print(forecasts)