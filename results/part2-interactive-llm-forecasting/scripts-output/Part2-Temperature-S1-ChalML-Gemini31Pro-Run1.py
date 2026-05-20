import pandas as pd
import numpy as np
import xgboost as xgb

# Set random seeds for reproducibility
np.random.seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
target_col = 'Daily minimum temperatures'
series = df[target_col].values

# Define split point based on Turn 0
train_size = 2920
test_size = 730
train_data = series[:train_size]
test_data = series[train_size:]

# Feature engineering: Lag features
lags = 14
def create_lags(data, n_lags):
    X, y = [], []
    for i in range(n_lags, len(data)):
        X.append(data[i-n_lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = create_lags(train_data, lags)

# Initialize and train XGBoost model with fixed hyperparameters from Turn 2
# multi_step_strategy 'N/A' is omitted
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)
model.fit(X_train, y_train)

# Recursive forecasting setup (Static strategy, no ground truth updates)
forecasts = []
# Initial window comes from the end of the training set
current_window = list(train_data[-lags:])

for i in range(test_size):
    # Prepare input features
    X_input = np.array(current_window).reshape(1, -1)
    # Predict one-step ahead
    pred = model.predict(X_input)[0]
    forecasts.append(float(pred))
    # Update window: remove oldest, append prediction (recursive)
    current_window.pop(0)
    current_window.append(pred)

# Output the final forecast list
print(forecasts)