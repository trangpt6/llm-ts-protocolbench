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

n = len(data)
train_size = int(0.8 * n)
train_data = data[:train_size]
test_data = data[train_size:]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_norm = (train_data - mean) / std

input_size = 2
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 2
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 50
batch_size = 16
lr = 0.001

X_list, Y_list = [], []
for i in range(len(train_norm) - seq_len - pred_len + 1):
    X_list.append(train_norm[i:i+seq_len])
    Y_list.append(train_norm[i+seq_len:i+seq_len+pred_len])
X_arr = np.array(X_list, dtype=np.float32)
Y_arr = np.array(Y_list, dtype=np.float32)

X_t = torch.tensor(X_arr).permute(0, 2, 1)
Y_t = torch.tensor(Y_arr).reshape(len(Y_arr), -1)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            ic = in_ch if i == 0 else out_ch
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(ic, out_ch, kernel_size, padding=padding, dilation=dilation))
            layers.append(Chomp1d(padding))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, kernel_size, dilations, num_layers, pred_len):
        super().__init__()
        blocks = []
        for i, d in enumerate(dilations):
            in_ch = input_size if i == 0 else hidden_size
            blocks.append(TemporalBlock(in_ch, hidden_size, kernel_size, d, num_layers))
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_size, output_size * pred_len)
        self.output_size = output_size
        self.pred_len = pred_len
    def forward(self, x):
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)

model = TCN(input_size, hidden_size, input_size, kernel_size, dilations, num_layers, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

ds = TensorDataset(X_t, Y_t)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

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
test_len = len(test_data)
buffer = [row.copy() for row in train_norm[-seq_len:]]

forecasts = []
with torch.no_grad():
    for step in range(test_len):
        x_in = np.array(buffer[-seq_len:], dtype=np.float32)
        x_in_t = torch.tensor(x_in).unsqueeze(0).permute(0, 2, 1).to(device)
        pred = model(x_in_t).cpu().numpy().reshape(pred_len, input_size)
        next_norm = pred[0]
        ice_pred = float(next_norm[1] * std[1] + mean[1])
        forecasts.append(ice_pred)
        buffer.append(next_norm.astype(np.float32))

print(forecasts)