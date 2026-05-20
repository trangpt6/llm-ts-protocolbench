import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
series = df['Daily minimum temperatures'].values.astype(np.float32)

# Split into train/test
train_size = 2921
train_series = series[:train_size]
test_size = len(series) - train_size

# Sliding window creation
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 14
X_train, y_train = create_sequences(train_series, seq_len)

# Reshape for TCN: (batch, input_size, seq_len)
X_train_t = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)  # (N, 1, seq_len)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# TCN model
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(dilations)
        in_channels = input_size
        for i in range(num_layers):
            dilation = dilations[i % num_levels]
            out_channels = hidden_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                        padding=(kernel_size-1)*dilation))
            in_channels = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(in_channels, output_size)

    def forward(self, x):
        # x: (batch, input_size, seq_len)
        y = self.tcn(x)  # (batch, hidden_size, seq_len')
        # take last time step output
        y = y[:, :, -1]  # (batch, hidden_size)
        return self.linear(y)

num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
model = TCN(input_size=1, output_size=1, hidden_size=64, num_layers=num_layers,
            kernel_size=kernel_size, dilations=dilations)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    # No validation printing to keep clean

# Recursive forecasting
model.eval()
forecasts = []
# Initial input window: last seq_len values of training data
input_seq = torch.tensor(train_series[-seq_len:][np.newaxis, np.newaxis, :], dtype=torch.float32).to(device)

with torch.no_grad():
    for _ in range(test_size):
        pred = model(input_seq)  # (1,1)
        next_val = pred.item()
        forecasts.append(next_val)
        # Update input sequence: remove first element, append predicted
        new_input = torch.cat([input_seq[:, :, 1:], pred.unsqueeze(2)], dim=2)
        input_seq = new_input

print(forecasts)