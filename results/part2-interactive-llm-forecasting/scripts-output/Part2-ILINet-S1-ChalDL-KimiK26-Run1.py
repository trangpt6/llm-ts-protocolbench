import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv(r'../../../data/ILINet.csv')

for col in ['AGE 25-49', 'AGE 50-64']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].ffill().bfill()

target = df['% WEIGHTED ILI'].values.astype(np.float32)

train_size = 1044
test_size = 262
train_data = target[:train_size]

seq_len = 52
pred_len = 1

def create_windows(data, seq_len):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = create_windows(train_data, seq_len)
X_train = torch.from_numpy(X_train).unsqueeze(-1)
y_train = torch.from_numpy(y_train).unsqueeze(-1)
X_train = X_train.permute(0, 2, 1)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilations[0])
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=0, dilation=dilations[1])
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = F.pad(x, ((self.kernel_size - 1) * self.dilations[0], 0))
        out = self.conv1(out)
        out = self.relu(out)
        out = F.pad(out, ((self.kernel_size - 1) * self.dilations[1], 0))
        out = self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        blocks = []
        in_ch = input_size
        for i in range(num_layers):
            d1 = dilations[2 * i]
            d2 = dilations[2 * i + 1]
            blocks.append(ResidualBlock(in_ch, hidden_size, kernel_size, [d1, d2]))
            in_ch = hidden_size
        self.network = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(input_size=1, hidden_size=64, num_layers=3, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32], pred_len=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(50):
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

model.eval()
history = list(train_data[-seq_len:])
forecasts = []
with torch.no_grad():
    for _ in range(test_size):
        x = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x).cpu().item()
        forecasts.append(pred)
        history.append(pred)

print(forecasts)