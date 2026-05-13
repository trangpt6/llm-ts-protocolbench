import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.sort_values('Month', inplace=True)
values = df['Passengers'].values.astype(np.float32)

# Chronological split: first 115 train, last 29 test (0-based)
train_size = 115
train_series = values[:train_size]
test_size = len(values) - train_size

# Hyperparameters
seq_len = 12
pred_len = 1
hidden_size = 64
num_layers = 2
epochs = 50
batch_size = 16
lr = 0.001
input_size = 1

# Create sliding window sequences for training
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len].reshape(seq_len, 1))
        y.append(data[i+seq_len].reshape(pred_len, 1))
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_series, seq_len)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=pred_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
model.train()
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Forecasting recursively over test set
model.eval()
# Initial sliding window: last seq_len points of train_series
current_seq = torch.tensor(train_series[-seq_len:].reshape(1, seq_len, 1), dtype=torch.float32).to(device)
forecasts = []
with torch.no_grad():
    for step in range(test_size):
        pred = model(current_seq).cpu().numpy().flatten()[0]
        forecasts.append(float(pred))
        # Update sequence: drop oldest, append predicted value
        new_val = torch.tensor([[[pred]]], dtype=torch.float32).to(device)
        current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)

print(forecasts)