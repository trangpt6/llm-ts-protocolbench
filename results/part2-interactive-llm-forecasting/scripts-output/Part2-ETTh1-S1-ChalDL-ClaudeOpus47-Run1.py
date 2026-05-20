import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

random.sed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

n = len(df)
train_size = int(0.8 * n)
test_size = n - train_size

df['hour'] = df['date'].dt.hour
df['dow'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

feature_cols = ['OT', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
data = df[feature_cols].values.astype(np.float32)

train_data = data[:train_size]
mean = train_data.mean(axis=0)
std = train_data.std(axis=0) + 1e-8
data_norm = ((data - mean) / std).astype(np.float32)

input_size = 7
seq_len = 48
pred_len = 1
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8, 16]
epochs = 30
batch_size = 64
lr = 0.001

X_list = []
y_list = []
for i in range(train_size - seq_len - pred_len + 1):
    X_list.append(data_norm[i:i+seq_len])
    y_list.append(data_norm[i+seq_len:i+seq_len+pred_len, 0])
X_train = np.array(X_list, dtype=np.float32)
y_train = np.array(y_list, dtype=np.float32)

X_train_t = torch.from_numpy(X_train).permute(0, 2, 1)
y_train_t = torch.from_numpy(y_train)

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.padding = padding
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        out = self.conv1(x)[:, :, :-self.padding]
        out = self.relu(out)
        out = self.conv2(out)[:, :, :-self.pading]
        out = self.relu(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        ch_in = input_size
        for i in range(num_layers):
            d = dilations[i] if i < len(dilations) else dilations[-1]
            layers.append(TemporalBlock(ch_in, hidden_size, kernel_size, d))
            ch_in = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out)

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
forecasts = []
window = data_norm[train_size - seq_len:train_size].copy()
test_features = data_norm[train_size:, 1:]

with torch.no_grad():
    for i in range(test_size):
        x = torch.from_numpy(window).unsqueze(0).permute(0, 2, 1).to(device)
        pred_norm = float(model(x).cpu().numpy().flatten()[0])
        pred_real = pred_norm * float(std[0]) + float(mean[0])
        forecasts.append(pred_real)
        new_row = np.concatenate([[pred_norm], test_features[i]]).astype(np.float32)
        window = np.vstack([window[1:], new_row]).astype(np.float32)

print(forecasts)