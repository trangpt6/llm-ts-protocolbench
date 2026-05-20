import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import random

# Set seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Preprocessing
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# Insert missing timestamp 2002-01-06
full_range = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='W-SUN')
df = df.set_index('DATE').reindex(full_range)
df = df.interpolate(method='linear')
df.index.name = 'DATE'
df = df.reset_index()

# Handle exogenous columns
df['AGE 25-49'] = df['AGE 25-49'].fillna(0)
df['AGE 50-64'] = df['AGE 50-64'].fillna(0)

target_col = '% WEIGHTED ILI'
data = df[target_col].values.reshape(-1, 1)

# Split
train_size = 1044
test_size = 261
train_data = data[:train_size]
test_data = data[train_size:]

# Scaling
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len]

# TCN Components
class ChainedCausalConv(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(ChainedCausalConv, self).__init__()
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.relu)
    def forward(self, x):
        return self.net(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [ChainedCausalConv(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.kernel_size = kernel_size

    def forward(self, x):
        # x shape: [batch, seq_len, input_size] -> [batch, input_size, seq_len]
        x = x.transpose(1, 2)
        y = self.network(x)
        # Slice to maintain causality (remove future padding)
        y = y[:, :, :x.size(2)]
        return self.linear(y[:, :, -1])

# Hyperparameters
hp = {'input_size': 1, 'seq_len': 52, 'pred_len': 1, 'hidden_size': 64, 'num_layers': 3, 'kernel_size': 3, 'dilations': [1, 2, 4, 8, 16, 32], 'epochs': 50, 'batch_size': 32, 'lr': 0.001}

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(hp['input_size'], hp['pred_len'], [hp['hidden_size']]*hp['num_layers'], hp['kernel_size']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'])
criterion = nn.MSELoss()

# Training
dataset = TimeSeriesDataset(train_scaled, hp['seq_len'])
dataloader = DataLoader(dataset, batch_size=hp['batch_size'], shuffle=True)

model.train()
for epoch in range(hp['epochs']):
    for seq, target in dataloader:
        seq, target = seq.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Recursive Forecasting
model.eval()
forecasts = []
current_seq = torch.FloatTensor(train_scaled[-hp['seq_len']:]).view(1, hp['seq_len'], 1).to(device)

with torch.no_grad():
    for _ in range(test_size):
        pred = model(current_seq)
        forecasts.append(pred.item())
        # Update sequence: remove first, append prediction
        new_val = pred.view(1, 1, 1)
        current_seq = torch.cat((current_seq[:, 1:, :], new_val), dim=1)

# Inverse scaling
final_forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten().tolist()
print(final_forecasts)