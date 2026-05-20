import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# No preprocessing
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

# Split indices
train_size = 158
test_size = 40

train_data = data[:train_size]
test_data = data[train_size:train_size+test_size]  # contains both Heater and Ice cream

# Create sequences for training
seq_len = 12
X_train = []
y_train = []
for i in range(len(train_data)-seq_len):
    X_train.append(train_data[i:i+seq_len])
    y_train.append(train_data[i+seq_len, 1])  # Ice cream is index 1

X_train = np.array(X_train)
y_train = np.array(y_train)

# Convert to tensors
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)

# Define TCN with fixed hyperparameters
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=(kernel_size-1)*dilation, dilation=dilation)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilations, seq_len, pred_len=1):
        super().__init__()
        self.seq_len = seq_len
        layers = []
        in_channels = input_size
        for dilation in dilations:
            layers.append(ResidualBlock(in_channels, hidden_size, kernel_size, dilation))
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size * seq_len, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out.reshape(out.size(0), -1)  # flatten
        out = self.fc(out)
        return out

# Hyperparameters (fixed)
input_size = 2
hidden_size = 32
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 50
batch_size = 16
lr = 0.001

model = TCN(input_size, hidden_size, kernel_size, dilations, seq_len)

# Training setup
dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # Optional: print(f'Epoch {epoch+1} loss: {total_loss/len(loader):.4f}')

# Recursive forecasting over test set
model.eval()
forecasts = []
# Start with last seq_len window from training data
window = train_data[-seq_len:].copy()  # shape (12,2)

with torch.no_grad():
    for i in range(test_size):
        # Prepare input: window as tensor (1, seq_len, input_size)
        x = torch.tensor(window).unsqueeze(0)  # (1, seq_len, 2)
        pred = model(x).item()
        forecasts.append(pred)
        # Update window: remove oldest, append new row with true Heater from test
        new_heater = test_data[i, 0]   # Heater
        new_row = np.array([new_heater, pred], dtype=np.float32)
        window = np.vstack([window[1:], new_row])

# Print the forecast list (Option 1)
print(forecasts)