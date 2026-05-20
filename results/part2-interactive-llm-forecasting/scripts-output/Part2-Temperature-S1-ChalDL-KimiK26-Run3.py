import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
values = df['Daily minimum temperatures'].values.astype(np.float32)

# train/test split
train_size = 2920
train_vals = values[:train_size]
test_vals = values[train_size:]

seq_len = 14
pred_len = 1
batch_size = 32
epochs = 50
lr = 0.001
kernel_size = 3
dilations = [1, 2, 4, 8]
hidden_size = 64
input_size = 1

# create sequences
def create_sequences(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_vals, seq_len)
X_train = torch.from_numpy(X_train).unsqueeze(-1)
y_train = torch.from_numpy(y_train).unsqueeze(-1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# TCN model
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + residual
        out = self.relu2(out)
        return out

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        in_ch = input_size
        for d in dilations:
            layers.append(ResidualBlock(in_ch, hidden_size, kernel_size, d))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out

device = torch.device('cpu')
model = TCN(input_size=input_size, hidden_size=hidden_size, kernel_size=kernel_size, dilations=dilations).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training
model.train()
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# recursive forecasting over test set
model.eval()
forecasts = []
history = train_vals[-seq_len:].tolist()

with torch.no_grad():
    for _ in range(len(test_vals)):
        x = torch.FloatTensor(history[-seq_len:]).view(1, seq_len, 1).to(device)
        pred = model(x).cpu().item()
        forecasts.append(pred)
        history.append(pred)

print(forecasts)