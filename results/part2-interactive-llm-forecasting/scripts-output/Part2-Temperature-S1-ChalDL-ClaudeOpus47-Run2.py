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

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)

full_idx = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
df = df.set_index('Date').reindex(full_idx)
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='linear')
series = df['Daily minimum temperatures'].values.astype(np.float32)

total = len(series)
train_size = int(0.8 * total)
train = series[:train_size]
test = series[train_size:]

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

train_mean = train.mean()
train_std = train.std()
train_norm = (train - train_mean) / train_std

X_list = []
y_list = []
for i in range(len(train_norm) - seq_len - pred_len + 1):
    X_list.append(train_norm[i:i+seq_len])
    y_list.append(train_norm[i+seq_len:i+seq_len+pred_len])
X_train = np.array(X_list, dtype=np.float32).reshape(-1, seq_len, input_size)
y_train = np.array(y_list, dtype=np.float32).reshape(-1, pred_len)

X_train_t = torch.from_numpy(X_train).permute(0, 2, 1)
y_train_t = torch.from_numpy(y_train)

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dilation):
        super().__init__()
        padding = (k - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, k, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=padding, dilation=dilation)
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
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        n_in = input_size
        for i in range(num_layers):
            d = dilations[i] if i < len(dilations) else dilations[-1]
            layers.append(TemporalBlock(n_in, hidden_size, kernel_size, d))
            n_in = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

model.train()
for ep in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

model.eval()
history = list(train_norm[-seq_len:])
forecasts_norm = []
with torch.no_grad():
    for _ in range(len(test)):
        x = np.array(history[-seq_len:], dtype=np.float32).reshape(1, input_size, seq_len)
        x_t = torch.from_numpy(x).to(device)
        out = model(x_t).cpu().numpy().flatten()[0]
        forecasts_norm.append(out)
        history.append(out)

forecasts = [float(v * train_std + train_mean) for v in forecasts_norm]
print(forecasts)