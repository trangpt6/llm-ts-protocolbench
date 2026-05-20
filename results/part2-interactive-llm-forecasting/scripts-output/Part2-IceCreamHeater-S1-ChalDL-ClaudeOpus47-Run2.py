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
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

total = len(df)
train_size = int(0.8 * total)
test_size = total - train_size

features = df[['Heater', 'Ice cream']].values.astype(np.float32)
target_idx = 1

train_data = features[:train_size]
test_data = features[train_size:]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
std[std == 0] = 1.0

train_norm = (train_data - mean) / std
test_norm = (test_data - mean) / std

seq_len = 12
pred_len = 1
input_size = 2
hidden_size = 32
num_layers = 2
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 50
batch_size = 16
lr = 0.001

X_train, y_train = [], []
for i in range(len(train_norm) - seq_len - pred_len + 1):
    X_train.append(train_norm[i:i+seq_len])
    y_train.append(train_norm[i+seq_len:i+seq_len+pred_len, target_idx])
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation)
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
        for i in range(num_layers):
            d = dilations[i] if i < len(dilations) else dilations[-1]
            in_ch = input_size if i == 0 else hidden_size
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size, d))
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.linear(out)

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

model.eval()
forecasts = []
history = [row.copy() for row in train_norm[-seq_len:]]
heater_test_norm = test_norm[:, 0]

with torch.no_grad():
    for t in range(test_size):
        window = np.array(history[-seq_len:], dtype=np.float32)
        x_in = torch.tensor(window).unsqueeze(0).to(device)
        pred_norm = float(model(x_in).cpu().numpy().flatten()[0])
        pred_real = pred_norm * std[target_idx] + mean[target_idx]
        forecasts.append(float(pred_real))
        next_vec = np.array([heater_test_norm[t], pred_norm], dtype=np.float32)
        history.append(next_vec)

print(forecasts)