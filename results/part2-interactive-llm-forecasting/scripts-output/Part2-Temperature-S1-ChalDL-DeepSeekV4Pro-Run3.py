import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Read the data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)
series = df['Daily minimum temperatures'].values.astype(np.float32)

# Chronological split: first 80% for training
train_size = 2921
train_series = series[:train_size]
test_series = series[train_size:]

# Normalize using training set
min_val = train_series.min()
max_val = train_series.max()
train_scaled = (train_series - min_val) / (max_val - min_val + 1e-8)
test_scaled = (test_series - min_val) / (max_val - min_val + 1e-8)

# Hyperparameters
seq_len = 14
pred_len = 1
input_size = 1
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
batch_size = 32
epochs = 50
lr = 0.001

# Create sequences for training
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, seq_len)
X_train = X_train.reshape(-1, seq_len, input_size).astype(np.float32)
y_train = y_train.reshape(-1, 1).astype(np.float32)

# Define TCN model
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        out = self.conv(x)
        # remove causal padding from the end
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(channels, channels, 1) if channels != channels else None

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return out + residual

class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, output_size):
        super(TCNModel, self).__init__()
        self.input_proj = nn.Conv1d(input_size, hidden_size, 1)
        layers = []
        for i in range(num_layers):
            layers.append(ResidualBlock(hidden_size, kernel_size, dilations[i % len(dilations)]))
        self.network = nn.Sequential(*layers)
        self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        x = self.input_proj(x)
        x = self.network(x)
        x = x.mean(dim=2)  # global average pooling over time
        return self.output_linear(x)

model = TCNModel(input_size=input_size, hidden_size=hidden_size,
                 num_layers=num_layers, kernel_size=kernel_size,
                 dilations=dilations, output_size=pred_len)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Prepare DataLoader
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
model.train()
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Recursive one-step forecasting on the test set
model.eval()
test_len = len(test_scaled)
input_seq = train_scaled[-seq_len:].copy()  # last seq_len points from training
input_seq = torch.tensor(input_seq, dtype=torch.float32).reshape(1, seq_len, input_size).to(device)

predictions_scaled = []
with torch.no_grad():
    for _ in range(test_len):
        pred = model(input_seq)  # output shape (1, pred_len)
        pred_value = pred.cpu().item()
        predictions_scaled.append(pred_value)
        # update input sequence: drop first, append predicted value
        new_value = torch.tensor(pred_value).reshape(1, 1, input_size).to(device)
        input_seq = torch.cat([input_seq[:, 1:, :], new_value], dim=1)

# Inverse transform to original scale
predictions = [p * (max_val - min_val + 1e-8) + min_val for p in predictions_scaled]

forecasts = predictions
print(forecasts)