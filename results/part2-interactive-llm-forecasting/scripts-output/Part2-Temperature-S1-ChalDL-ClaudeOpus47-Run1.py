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

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

values = df['Daily minimum temperatures'].astype(float).values

n = len(values)
train_size = int(0.8 * n)
train = values[:train_size]
test = values[train_size:]

input_size = 1
seq_len = 14
pred_len = 1
hidden_size = 64
num_layers = 3
epochs = 50
batch_size = 32
lr = 0.001
kernel_size = 3
dilations = [1, 2, 4, 8]

mean = float(train.mean())
std = float(train.std())
train_n = (train - mean) / std

X_list = []
y_list = []
for i in range(len(train_n) - seq_len - pred_len + 1):
    X_list.append(train_n[i:i + seq_len])
    y_list.append(train_n[i + seq_len:i + seq_len + pred_len])

X_arr = np.array(X_list, dtype=np.float32)
y_arr = np.array(y_list, dtype=np.float32)

X_tensor = torch.from_numpy(X_arr).unsqueeze(1)
y_tensor = torch.from_numpy(y_arr)

dataset = TensorDataset(X_tensor, y_tensor)
g = torch.Generator()
g.manual_seed(42)
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
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        used = dilations[:num_layers]
        layers = []
        in_ch = input_size
        for d in used:
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size, d))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out)

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

model.train()
for _ in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

model.eval()
history = list(train_n[-seq_len:])
forecasts = []
with torch.no_grad():
    for _ in range(len(test)):
        x_in = np.array(history[-seq_len:], dtype=np.float32).reshape(1, 1, seq_len)
        x_t = torch.from_numpy(x_in).to(device)
        p = float(model(x_t).cpu().numpy().flatten()[0])
        forecasts.append(p * std + mean)
        history.append(p)

print(forecasts)