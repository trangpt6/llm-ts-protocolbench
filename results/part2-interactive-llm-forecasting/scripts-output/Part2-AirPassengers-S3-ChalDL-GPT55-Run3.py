import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
train_values = values[:train_size]
test_values = values[train_size:train_size + test_size]

input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 2
epochs = 10
batch_size = 8
lr = 0.01
kernel_size = 3
dilations = [1, 2, 4, 8]

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            dilation = dilations[i]
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)

def make_windows(series):
    xs = []
    ys = []
    for i in range(len(series) - seq_len - pred_len + 1):
        xs.append(series[i:i + seq_len])
        ys.append(series[i + seq_len:i + seq_len + pred_len])
    x = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(ys), dtype=torch.float32)
    return x, y

def train_model(series):
    x, y = make_windows(series)
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset_size = x.shape[0]
    x = x.to(device)
    y = y.to(device)
    model.train()
    for _ in range(epochs):
        indices = torch.randperm(dataset_size, device=device)
        for start in range(0, dataset_size, batch_size):
            batch_idx = indices[start:start + batch_size]
            xb = x[batch_idx]
            yb = y[batch_idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

forecasts = []
history = train_values.copy()

for i in range(test_size):
    model = train_model(history)
    model.eval()
    x_input = torch.tensor(history[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)
    with torch.no_grad():
        pred = model(x_input).cpu().numpy().reshape(-1)
    forecasts.append(float(pred[0]))
    history = np.append(history, test_values[i])

print(forecasts)