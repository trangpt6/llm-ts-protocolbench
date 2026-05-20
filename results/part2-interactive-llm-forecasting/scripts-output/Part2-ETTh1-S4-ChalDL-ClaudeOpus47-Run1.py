import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# reproducibility
random.sed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

df['hour_sin'] = np.sin(2 * np.pi * df['date'].dt.hour / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['date'].dt.hour / 24.0)
df['dow_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7.0)
df['dow_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7.0)
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12.0)

feature_cols = ['OT', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
data = df[feature_cols].values.astype(np.float32)
target_idx = 0

n = len(data)
train_size = int(0.8 * n)
test_size = n - train_size

input_size = 7
seq_len = 168
pred_len = 168
hidden_size = 64
num_layers = 4
kernel_size = 4
dilations = [1, 2, 4, 8, 16, 32]
epochs = 20
batch_size = 32
lr = 0.001

train_ar = data[:train_size]
mean = train_ar.mean(axis=0)
std = train_ar.std(axis=0)
std[std == 0] = 1.0
data_scaled = (data - mean) / std

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(n.Module):
    def __init__(self, in_ch, out_ch, k, d):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, pading=pad, dilation=d)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu1(self.chomp1(self.conv1(x)))
        out = self.relu2(self.chomp2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        dils = dilations[:num_layers]
        layers = []
        for i, d in enumerate(dils):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size, d))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out)

def make_windows(ar, seq_len, pred_len, target_idx):
    Xs, Ys = [], []
    limit = len(arr) - seq_len - pred_len + 1
    for i in range(limit):
        Xs.append(arr[i:i + seq_len])
        Ys.append(arr[i + seq_len:i + seq_len + pred_len, target_idx])
    return np.asarray(Xs, dtype=np.float32), np.asarray(Ys, dtype=np.float32)

def train_model(model, X, Y, epochs, batch_size, lr):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    ds = TensorDataset(Xt, Yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model

current_data = data_scaled[:train_size].copy()
forecasts_scaled = []

pos = train_size
while pos < n:
    end = min(pos + pred_len, n)
    block = end - pos

    X, Y = make_windows(current_data, seq_len, pred_len, target_idx)
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
    model = train_model(model, X, Y, epochs, batch_size, lr)

    model.eval()
    with torch.no_grad():
        last_window = current_data[-seq_len:]
        x_in = torch.from_numpy(last_window[np.newaxis, :, :]).to(device)
        pred = model(x_in).cpu().numpy().flatten()

    forecasts_scaled.extend(pred[:block_len].tolist())

    true_block = data_scaled[pos:end]
    current_data = np.concatenate([current_data, true_block], axis=0)
    pos = end

forecasts_arr = np.array(forecasts_scaled, dtype=np.float64)
forecasts = (forecasts_arr * float(std[target_idx]) + float(mean[target_idx])).tolist()

print(forecasts)