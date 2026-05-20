import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use CPU if CUDA is unavailable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 2
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 30
batch_size = 16
lr = 0.001

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

train_size = 158
test_size = 40
feature_cols = ["Heater", "Ice cream"]
target_col = "Ice cream"

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
            dilation = dilations[i]
            in_channels = input_size if i == 0 else hidden_size
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.head(y)

def make_windows(data_values):
    X, y = [], []
    target_idx = feature_cols.index(target_col)
    for i in range(0, len(data_values) - seq_len - pred_len + 1):
        X.append(data_values[i:i + seq_len, :])
        y.append(data_values[i + seq_len:i + seq_len + pred_len, target_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

def train_model(history_df):
    values = history_df[feature_cols].values.astype(np.float32)
    X, y = make_windows(values)
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model

forecasts = []
history = df.iloc[:train_size].copy()
test = df.iloc[train_size:train_size + test_size].copy()
start = 0

while start < test_size:
    block_len = min(12, test_size - start)
    model = train_model(history)
    x_input = history[feature_cols].values.astype(np.float32)[-seq_len:]
    x_tensor = torch.tensor(x_input.reshape(1, seq_len, input_size), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy().reshape(-1)
    forecasts.extend(pred[:block_len].astype(float).tolist())
    history = pd.concat([history, test.iloc[start:start + block_len]], axis=0).reset_index(drop=True)
    start += block_len

print(forecasts)