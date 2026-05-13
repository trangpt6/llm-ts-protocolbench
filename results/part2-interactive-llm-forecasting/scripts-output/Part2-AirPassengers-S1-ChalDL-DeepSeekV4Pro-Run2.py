import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset
data = pd.read_csv(r'../../../data/AirPassengers.csv')
series = data['Passengers'].values.astype(np.float32)

# Chronological 80/20 split
train_size = 115
train = series[:train_size]
test = series[train_size:]

seq_len = 12

# Create sliding windows from training data
X, y = [], []
for i in range(len(train) - seq_len):
    X.append(train[i:i+seq_len])
    y.append(train[i+seq_len])
X = np.array(X).reshape(-1, seq_len, 1)
y = np.array(y).reshape(-1, 1)

X_t = torch.tensor(X, dtype=torch.float32).to(device)
y_t = torch.tensor(y, dtype=torch.float32).to(device)

# LSTM model definition
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
        self.linear = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (no shuffling, sequential batches)
epochs = 50
batch_size = 16
num_samples = X_t.shape[0]
model.train()
for epoch in range(epochs):
    for i in range(0, num_samples, batch_size):
        end = min(i+batch_size, num_samples)
        xb = X_t[i:end]
        yb = y_t[i:end]
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

# Recursive one-step ahead forecast over test period
model.eval()
last_window = torch.tensor(train[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)
forecasts = []
with torch.no_grad():
    for _ in range(len(test)):
        pred = model(last_window)
        forecasts.append(pred.item())
        last_window = torch.cat((last_window[:, 1:, :], pred.view(1, 1, 1)), dim=1)

print(forecasts)