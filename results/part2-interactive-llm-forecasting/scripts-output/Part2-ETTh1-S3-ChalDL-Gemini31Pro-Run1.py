import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import random

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Feature engineering to match input_size=7
df['hour'] = df.index.hour
df['day'] = df.index.day
df['weekday'] = df.index.weekday
df['month'] = df.index.month
df['dayofyear'] = df.index.dayofyear
df['is_weekend'] = (df.index.weekday >= 5).astype(float)

# Reorder columns with OT at the end
cols = ['hour', 'day', 'weekday', 'month', 'dayofyear', 'is_weekend', 'OT']
df = df[cols]
data = df.values

# Train test split
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

# Scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Hyperparameters
input_size = 7
seq_len = 12
pred_len = 24
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

# Dataset creation
def create_dataset(dataset):
    X, y = [], []
    for i in range(len(dataset) - seq_len - pred_len + 1):
        X.append(dataset[i : i + seq_len])
        y.append(dataset[i + seq_len : i + seq_len + pred_len, -1])
    return np.array(X), np.array(y)

# Model definition
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Initial training
train_data = data_scaled[:train_size]
X_train, y_train = create_dataset(train_data)
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Forecasting
forecasts = []
history_data = list(data_scaled[:train_size])

for t in range(train_size, total_timesteps):
    model.eval()
    x_input = history_data[-seq_len:]
    x_tensor = torch.tensor(np.array([x_input]), dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x_tensor)
    pred = pred.cpu().numpy()[0]
    
    dummy = np.zeros((pred_len, input_size))
    dummy[:, -1] = pred
    pred_unscaled = scaler.inverse_transform(dummy)[:, -1]
    
    forecasts.extend(pred_unscaled.tolist())
    
    true_val = data_scaled[t]
    history_data.append(true_val)
    
    # Retraining on the rolling window
    current_history = np.array(history_data[-train_size:])
    X_train_new, y_train_new = create_dataset(current_history)
    
    X_t = torch.tensor(X_train_new, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train_new, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

print(forecasts)