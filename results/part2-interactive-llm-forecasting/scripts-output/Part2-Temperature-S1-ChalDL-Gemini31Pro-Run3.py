import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import random

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv')
target_col = 'Daily minimum temperatures'
data = df[target_col].values.astype(np.float32).reshape(-1, 1)

# Split data
train_size = 2921
test_size = 731
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
hp = {
    'input_size': 1,
    'seq_len': 14,
    'pred_len': 1,
    'hidden_size': 64,
    'num_layers': 3,
    'epochs': 50,
    'batch_size': 32,
    'lr': 0.001,
    'kernel_size': 3,
    'dilations': [1, 2, 4, 8]
}

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return (self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len])

# TCN Model components
class ChausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ChausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = ChausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = ChausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = hp['dilations'][i]
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size, dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        y1 = self.network(x.transpose(1, 2))
        return self.linear(y1[:, :, -1])

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(hp['input_size'], hp['pred_len'], [hp['hidden_size']]*hp['num_layers'], hp['kernel_size'], 0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
criterion = nn.MSELoss()

train_dataset = TimeSeriesDataset(train_data, hp['seq_len'])
train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)

model.train()
for epoch in range(hp['epochs']):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Recursive Inference
model.eval()
forecasts = []
# Initial window is the last seq_len observations of the training set
current_window = train_data[-hp['seq_len']:].reshape(1, hp['seq_len'], 1)
current_window = torch.from_numpy(current_window).to(device)

with torch.no_grad():
    for _ in range(test_size):
        pred = model(current_window)
        pred_val = pred.item()
        forecasts.append(pred_val)
        # Update window: remove first, append prediction
        new_val = torch.tensor([[[pred_val]]], dtype=torch.float32).to(device)
        current_window = torch.cat((current_window[:, 1:, :], new_val), dim=1)

print(forecasts)