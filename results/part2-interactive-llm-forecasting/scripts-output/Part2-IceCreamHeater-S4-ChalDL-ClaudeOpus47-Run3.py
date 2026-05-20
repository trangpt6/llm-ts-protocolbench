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

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)
target_idx = 1

n = len(data)
train_size = int(0.8 * n)
test_size = n - train_size

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
block_size = 12

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(in_ch, out_ch, k, padding=self.pad, dilation=d)
    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]
        return out

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = input_size
        for d in dilations:
            layers.append(CausalConv1d(in_ch, hidden_size, kernel_size, d))
            layers.append(nn.ReLU())
            in_ch = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.fc(out)

def make_windows(arr):
    X, Y = [], []
    for i in range(len(arr) - seq_len - pred_len + 1):
        X.append(arr[i:i+seq_len])
        Y.append(arr[i+seq_len:i+seq_len+pred_len, target_idx])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_model(train_data):
    # deterministic init per retrain
    torch.manual_seed(42)
    model = TCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X, Y = make_windows(train_data)
    if len(X) == 0:
        return model
    X_t = torch.tensor(X).to(device)
    Y_t = torch.tensor(Y).to(device)
    ds = TensorDataset(X_t, Y_t)
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

forecasts = []
current_data = data[:train_size].copy()
n_blocks = (test_size + block_size - 1) // block_size

for b in range(n_blocks):
    model = train_model(current_data)
    model.eval()
    last_seq = current_data[-seq_len:]
    x_in = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_in).cpu().numpy().flatten()
    block_start = train_size + b * block_size
    block_end = min(block_start + block_size, n)
    actual_size = block_end - block_start
    forecasts.extend(pred[:actual_size].tolist())
    current_data = np.vstack([current_data, data[block_start:block_end]])

print(forecasts)