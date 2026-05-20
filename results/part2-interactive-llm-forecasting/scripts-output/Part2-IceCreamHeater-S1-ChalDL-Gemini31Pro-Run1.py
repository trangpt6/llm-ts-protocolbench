import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']
data = df[feature_cols].values.astype(float)

# Split data
train_size = 158
test_size = 40
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
seq_len = 12
input_size = 2
hidden_size = 32
num_layers = 2
kernel_size = 3
dilations = [1, 2, 4, 8]
lr = 0.001
epochs = 50
batch_size = 16

# Scaling
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_scaled = (train_data - mean) / std

# Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len, 1:2]

train_dataset = TimeSeriesDataset(train_scaled, seq_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# TCN Model
class ChausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        res = self.res(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x + res

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super().__init__()
        layers = []
        curr_in = input_size
        for i in range(num_layers):
            for d in dilations:
                layers.append(TCNBlock(curr_in, hidden_size, kernel_size, d))
                curr_in = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        return self.fc(x[:, :, -1])

device = torch.device('cpu')
model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
model.train()
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

# Inference (Recursive)
model.eval()
forecasts = []
current_seq = train_scaled[-seq_len:]

for i in range(test_size):
    with torch.no_grad():
        input_tensor = torch.FloatTensor(current_seq).unsqueeze(0).to(device)
        pred_scaled = model(input_tensor).item()
        
        # Inverse scaling for the target (Ice cream is index 1)
        pred_unscaled = pred_scaled * std[1] + mean[1]
        forecasts.append(pred_unscaled)
        
        # Update sequence: use prediction for target, and for multivariate 'Heater' we must assume a value.
        # Since ground truth is disabled and no future covariates provided, we use the last known 'Heater' value.
        next_step = np.array([[current_seq[-1, 0], pred_scaled]])
        current_seq = np.append(current_seq[1:], next_step, axis=0)

print(forecasts)