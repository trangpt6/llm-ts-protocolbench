import pandas as pd
import numpy as np
import lightgbm as lgb
import random
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Load and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index.freq = 'D'

target = 'Daily minimum temperatures'
series = df[target].values

# Train/test split
train_size = int(0.8 * len(series))
train_series = series[:train_size]
test_series = series[train_size:]

# Feature engineering function
def create_features(data, lags):
    X, y = [], []
    for i in range(len(data) - lags):
        X.append(data[i:i+lags])
        y.append(data[i+lags])
    return np.array(X), np.array(y)

lags = 7
forecasts = []
history = list(train_series)

# Rolling forecast with retraining
for t in range(len(test_series)):
    X_train, y_train = create_features(history, lags)
    
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    X_test = np.array(history[-lags:]).reshape(1, -1)
    y_pred = model.predict(X_test)[0]
    forecasts.append(float(y_pred))
    
    history.append(test_series[t])

print(forecasts)