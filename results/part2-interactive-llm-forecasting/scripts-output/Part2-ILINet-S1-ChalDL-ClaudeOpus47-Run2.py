import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'
mask_nonreport = (df['TOTAL PATIENTS'] == 0) & (df['NUM. OF PROVIDERS'] == 0)
df.loc[mask_nonreport, target_col] = np.nan
df[target_col] = df[target_col].interpolate(method='linear', limit_direction='both')

series = df[target_col].values.astype(np.float32)

n = len(series)
train_size = int(0.8 * n)
train = series[:train_size]
test = series[train_size:]

input_size = 1
seq_len = 52
pred_len = 1
hidden_size = 64
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32]
epochs = 50
batch_size = 32
lr = 0.001

mean = float(train.mean())
std = float(train.std() + 1e-8)
train_norm = (train - mean) / std

def make_sequences(arr, sl, pl):
    X, y = [], []
    for i in range(len(arr) - sl - pl + 1):
        X.append(arr[i:i+sl])
        y.append(arr[i+sl:i+sl+pl])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = make_sequences(train_norm, seq_len, pred_len)
X_train = X_train.reshape(-1, input_size, seq_len)
y_train = y_train.reshape(-1, pred_len)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

# Reproducible loader
g = torch.Generator()
g.manual_seed(42)
dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, d):
        super().__init__()
        padding = (k - 1) * d
        self.conv1 = nn.Conv1d(n_in, n_out, k, padding=padding, dilation=d)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=padding, dilation=d)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_size, hid, k, dils, pl):
        super().__init__()
        layers = []
        ch_in = in_size
        for d in dils:
            layers.append(TemporalBlock(ch_in, hid, k, d))
            ch_in = hid
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hid, pl)
    def forward(self, x):
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.fc(out)

model = TCN(input_size, hidden_size, kernel_size, dilations, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

model.eval()

history_norm = list(train_norm[-seq_len:].astype(np.float32))
forecasts = []
with torch.no_grad():
    for _ in range(len(test)):
        window = np.array(history_norm[-seq_len:], dtype=np.float32).reshape(1, input_size, seq_len)
        x_in = torch.tensor(window, dtype=torch.float32).to(device)
        pred_norm = float(model(x_in).cpu().numpy().flatten()[0])
        pred_val = pred_norm * std + mean
        forecasts.append(float(pred_val))
        history_norm.append(pred_norm)

print(forecasts)