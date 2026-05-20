import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values

# Train/test split
total_timesteps = len(target)
train_size = int(0.8 * total_timesteps)
train_data = target[:train_size]
test_data = target[train_size:]

# Hyperparameters
input_size = 1
seq_len = 14
pred_len = 1
hidden_size = 64
num_layers = 3
epochs = 50
batch_size = 32
lr = 0.001
kernel_size = 3
dilations = [1, 2, 4, 8][:num_layers]

# Prepare training sequences
X_train, y_train = [], []
for i in range(len(train_data) - seq_len):
    X_train.append(train_data[i:i+seq_len])
    y_train.append(train_data[i+seq_len])

X_train = np.array(X_train).reshape(-1, seq_len, input_size)
y_train = np.array(y_train).reshape(-1, pred_len)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define TCN components
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dilations):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(dilations)
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1)*dilation_size)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dilations):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dilations)
        self.linear = nn.Linear(num_channels, output_size)
    def forward(self, x):
        x = x.transpose(1, 2)
        y1 = self.tcn(x)
        out = self.linear(y1[:, :, -1])
        return out

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(input_size=input_size, output_size=pred_len, num_channels=hidden_size, kernel_size=kernel_size, dilations=dilations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(epochs):
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Recursive forecasting
model.eval()
forecasts = []
current_seq = train_data[-seq_len:].tolist()

with torch.no_grad():
    for _ in range(len(test_data)):
        seq_tensor = torch.tensor(current_seq, dtype=torch.float32).reshape(1, seq_len, input_size).to(device)
        pred = model(seq_tensor).item()
        forecasts.append(pred)
        current_seq.append(pred)
        current_seq.pop(0)

print(forecasts)