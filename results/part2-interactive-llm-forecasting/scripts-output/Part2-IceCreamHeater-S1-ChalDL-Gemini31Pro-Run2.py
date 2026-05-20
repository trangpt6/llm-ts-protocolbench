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
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        y1 = self.network(x.transpose(1, 2))
        return self.linear(y1[:, :, -1])

# Dataset class
class TSData(Dataset):
    def __init__(self, data, seq_len):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+self.seq_len, 1]

# Load and split data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(float)
train_size = 158
test_size = 40
train_data = data[:train_size]

# Hyperparameters
seq_len = 12
hidden_size = 32
num_layers = 2
kernel_size = 3
epochs = 50
batch_size = 16
lr = 0.001

# Prepare training
dataset = TSData(train_data, seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = TCN(input_size=2, num_channels=[hidden_size]*num_layers, kernel_size=kernel_size)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training loop
model.train()
for epoch in range(epochs):
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.squeeze(), y_batch)
        loss.backward()
        optimizer.step()

# Inference (Static, Recursive)
model.eval()
forecasts = []
# Initial window from end of training set
current_window = train_data[-seq_len:].tolist()

for i in range(test_size):
    input_tensor = torch.FloatTensor([current_window])
    with torch.no_grad():
        pred = model(input_tensor).item()
    forecasts.append(pred)
    
    # Update window: use predicted Ice cream and assume Heater follows a simple cycle or last value
    # Since Heater is exogenous and we have no ground truth, we use the last known Heater value
    # or the value from the test set if we treat it as a known future covariate.
    # Given "no updates using true values", we use the test set's Heater values as they are covariates.
    next_heater = data[train_size + i, 0]
    current_window.pop(0)
    current_window.append([next_heater, pred])

print(forecasts)