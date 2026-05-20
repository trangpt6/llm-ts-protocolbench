import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
series = df['Passengers'].values.astype(np.float32)

n = len(series)
train_size = int(0.8 * n)
n_test = n - train_size

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
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, d):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(n_in, n_out, k, padding=pad, dilation=d)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=pad, dilation=d)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.out_relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_size, hid, n_layers, k, dils, pred):
        super().__init__()
        layers = []
        chans = [in_size] + [hid] * n_layers
        for i in range(n_layers):
            d = dils[i % len(dils)]
            layers.append(TemporalBlock(chans[i], chans[i+1], k, d))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hid, pred)
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)

def make_windows(arr, sl, pl):
    X, Y = [], []
    for i in range(len(arr) - sl - pl + 1):
        X.append(arr[i:i+sl])
        Y.append(arr[i+sl:i+sl+pl])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

forecasts = []

for step in range(n_test):
    available = series[:train_size + step].copy()
    mu = float(available.mean())
    sd = float(available.std() + 1e-8)
    norm = (available - mu) / sd

    X, Y = make_windows(norm, seq_len, pred_len)
    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)
    Y_t = torch.from_numpy(Y).to(device)

    ds = TensorDataset(X_t, Y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    torch.manual_seed(42 + step)
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        last_seq = norm[-seq_len:]
        x_in = torch.from_numpy(last_seq).unsqueeze(0).unsqueeze(-1).to(device)
        out = model(x_in).cpu().numpy().flatten()
        out = out * sd + mu

    forecasts.append(float(out[0]))

print(forecasts)