import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# TCN Model Definition
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
            dilation_size = [1, 2, 4, 8][i]
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size, dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        y1 = self.network(x.transpose(1, 2))
        return self.linear(y1[:, :, -1])

# Dataset preparation
class TSData(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len, 1]

# Load and split data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(float)
train_size = 158
test_size = 40
seq_len = 12
pred_len = 12
batch_size = 16
epochs = 30
lr = 0.001

# Scaling
mean = data[:train_size].mean(axis=0)
std = data[:train_size].std(axis=0)
data_scaled = (data - mean) / std

def train_model(train_data_scaled):
    dataset = TSData(train_data_scaled, seq_len, pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TCN(2, 12, [32, 32, 32], 3, 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    return model

all_forecasts = []
current_train_end = train_size

# Block-wise rolling update
while len(all_forecasts) < test_size:
    # Retrain model
    current_train_data = data_scaled[:current_train_end]
    model = train_model(current_train_data)
    
    # Forecast next block
    model.eval()
    with torch.no_grad():
        last_seq = torch.FloatTensor(current_train_data[-seq_len:]).unsqueeze(0)
        pred = model(last_seq).numpy().flatten()
        # Inverse scale target (Ice cream is index 1)
        pred_inv = pred * std[1] + mean[1]
        
        # Determine how many steps to take from this block
        remaining = test_size - len(all_forecasts)
        take = min(pred_len, remaining)
        all_forecasts.extend(pred_inv[:take].tolist())
        
        # Update training end for next block (ground truth enabled)
        current_train_end += take

print(all_forecasts)