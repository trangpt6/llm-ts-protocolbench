import random
import numpy as np
import pandas as pd
import lightgbm as lgb

random.seed(42)
np.random.seed(42)

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])

# Preprocessing: create full daily date range and forward fill missing dates
full_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
df = df.set_index('Date').reindex(full_dates, method='ffill').reset_index()
df.columns = ['Date', 'Daily minimum temperatures']

# Chronological split
train = df.iloc[:2920]
test = df.iloc[2920:]

train_values = train['Daily minimum temperatures'].tolist()
data = train_values.copy()
forecasts = []

for i in range(len(test)):
    true_val = test.iloc[i]['Daily minimum temperatures']
    
    # Prepare training data using lags of 7
    X_train = []
    y_train = []
    for j in range(7, len(data)):
        X_train.append(data[j-7:j])
        y_train.append(data[j])
    
    # Last 7 values for prediction
    X_pred = [data[-7:]]
    
    # Train model
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(np.array(X_train), np.array(y_train))
    
    # Predict next step
    pred = model.predict(np.array(X_pred))[0]
    forecasts.append(pred)
    
    # Append true value to data for next iteration
    data.append(true_val)

print(forecasts)