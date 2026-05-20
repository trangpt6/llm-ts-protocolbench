import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]

train_size = 158
test_size = 40
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:train_size + test_size].copy()

params = {'input_size': 2, 'seq_len': 12, 'pred_len': 12, 'hidden_size': 32, 'num_layers': 3, 'kernel_size': 3, 'dilations': [1, 2, 4, 8], 'epochs': 30, 'batch_size': 16, 'lr': 0.001}

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
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = input_size if i == 0 else hidden_size
            dilation = dilations[i % len(dilations)]
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size, dilation))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)

def make_sequences(data_array, target_array, seq_len, pred_len):
    X, y = [], []
    n = len(data_array)
    for i in range(n - seq_len - pred_len + 1):
        X.append(data_array[i:i + seq_len])
        y.append(target_array[i + seq_len:i + seq_len + pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(history_df):
    values = history_df[feature_cols].values.astype(np.float32)
    target = history_df[target_col].values.astype(np.float32)
    mean_x = values.mean(axis=0)
    std_x = values.std(axis=0)
    std_x[std_x == 0] = 1.0
    mean_y = target.mean()
    std_y = target.std()
    if std_y == 0:
        std_y = 1.0
    values_scaled = (values - mean_x) / std_x
    target_scaled = (target - mean_y) / std_y
    X, y = make_sequences(values_scaled, target_scaled, params["seq_len"], params["pred_len"])
    model = TCN(params["input_size"], params["hidden_size"], params["num_layers"], params["kernel_size"], params["dilations"], params["pred_len"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)
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
    return model, mean_x, std_x, mean_y, std_y

def forecast_block(model, history_df, mean_x, std_x, mean_y, std_y):
    last_x = history_df[feature_cols].values.astype(np.float32)[-params["seq_len"]:]
    last_x = (last_x - mean_x) / std_x
    x_tensor = torch.tensor(last_x[None, :, :], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_tensor).cpu().numpy().ravel()
    pred = pred_scaled * std_y + mean_y
    return pred.tolist()

history = train_df.copy()
forecasts = []
start = 0

while start < test_size:
    model, mean_x, std_x, mean_y, std_y = train_model(history)
    block_pred = forecast_block(model, history, mean_x, std_x, mean_y, std_y)
    remaining = test_size - start
    take = min(params["pred_len"], remaining)
    forecasts.extend([float(x) for x in block_pred[:take]])
    history = pd.concat([history, test_df.iloc[start:start + take]], ignore_index=True)
    start += take

print(forecasts)