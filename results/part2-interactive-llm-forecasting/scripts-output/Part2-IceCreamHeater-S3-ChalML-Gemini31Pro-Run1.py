import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Set random seed for reproducibility
np.random.seed(42)

# Read data and apply preprocessing
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df = df.sort_index()

# Extract features
data = df[['Heater', 'Ice cream']].values

train_size = 158
total_timesteps = len(data)
lags = 6
horizon = 12

all_forecasts = []

for step in range(train_size, total_timesteps):
    train_data = data[:step]
    
    X_train = []
    y_train = []
    for i in range(lags, len(train_data)):
        X_train.append(train_data[i-lags:i].flatten())
        y_train.append(train_data[i])
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    current_seq = train_data[-lags:].copy()
    
    for h in range(horizon):
        pred = model.predict(current_seq.flatten().reshape(1, -1))[0]
        all_forecasts.append(float(pred[1]))
        current_seq = np.vstack([current_seq[1:], pred])

print(all_forecasts)