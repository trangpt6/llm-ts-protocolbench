import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {'input_size': 1, 'seq_len': 12, 'pred_len': 12, 'hidden_size': 32, 'num_layers': 2, 'epochs': 10, 'batch_size': 8, 'lr': 0.01, 'kernel_size': 3, 'dilations': [1, 2, 4, 8]}

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df["Passengers"].astype(float).to_numpy()

train_size = 115
test_size = 29
train_values = values[:train_size]
test_values = values[train_size:train_size + test_size]

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
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, pred_len, kernel_size, dilations, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilations[i % len(dilations)]
            in_channels = input_size if i == 0 else hidden_size
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)

def make_sequences(series, seq_len, pred_len):
    xs = []
    ys = []
    for i in range(len(series) - seq_len - pred_len + 1):
        xs.append(series[i:i + seq_len])
        ys.append(series[i + seq_len:i + seq_len + pred_len])
    x = np.array(xs, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(ys, dtype=np.float32).reshape(-1, pred_len)
    return x, y

forecasts = []

for step in range(test_size):
    # Reset seeds before each retraining for reproducibility
    random.seed(42 + step)
    np.random.seed(42 + step)
    torch.manual_seed(42 + step)

    available = values[:train_size + step].astype(float)
    mean = available.mean()
    std = available.std()
    if std == 0:
        std = 1.0
    scaled_available = (available - mean) / std

    x_train, y_train = make_sequences(scaled_available, params["seq_len"], params["pred_len"])
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=params["batch_size"], shuffle=False)

    model = TCN(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        pred_len=params["pred_len"],
        kernel_size=params["kernel_size"],
        dilations=params["dilations"],
        num_layers=params["num_layers"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(params["epochs"]):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    last_window = scaled_available[-params["seq_len"]:].astype(np.float32).reshape(1, params["seq_len"], 1)
    x_input = torch.tensor(last_window, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_input).cpu().numpy().reshape(-1)

    pred_unscaled = pred_scaled * std + mean
    forecasts.append(float(pred_unscaled[0]))

print(forecasts)